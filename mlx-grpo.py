import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_map
from datasets import load_dataset, Dataset
from mlx_lm import load as mlx_load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.utils import load_model as mlx_load_model_only
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Tuple
import math
from mlx.optimizers import Adam, SGD, cosine_decay, clip_grad_norm
import re
import copy
import inspect
import random
import argparse
import tomllib  # Python 3.11+
import pickle
from pathlib import Path

# -------------------------------------------------------------------
# Dataset Preparation and Formatting
# -------------------------------------------------------------------
SYSTEM_PROMPT = """
Respond in the following format:

<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# Uncomment the middle messages below for 1-shot prompting if desired.
def get_gsm8k_questions(split="train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split]  # type: ignore
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            # {'role': 'user', 'content': 'What is the largest single-digit prime number?'},
            # {'role': 'assistant', 'content': XML_COT_FORMAT.format(
            #     reasoning="9 is divisible by 3 and 8 is divisible by 2, but 7 is prime.",
            #     answer="7"
            # )},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })  # type: ignore
    return data  # type: ignore

# -------------------------------------------------------------------
# Reward Functions
# -------------------------------------------------------------------
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    gold = answer[0] if isinstance(answer, (list, tuple)) else answer
    if gold is None:
        return [0.0] * len(extracted_responses)
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{gold}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == gold else 0.0 for r in extracted_responses]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    scores = []
    for r in extracted_responses:
        try:
            _ = int(r.strip())
            scores.append(0.5)
        except Exception:
            scores.append(0.0)
    return scores

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    # Require full XML shell; allow arbitrary newlines/whitespace
    pattern = r"^\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>\s*$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    score = 0.0
    if re.search(r"<reasoning>.*?</reasoning>", text, flags=re.DOTALL):
        score += 0.25
    if re.search(r"<answer>.*?</answer>", text, flags=re.DOTALL):
        score += 0.25
    # Penalize trailing junk after </answer>
    end = re.search(r"</answer>(.*)$", text, flags=re.DOTALL)
    if end:
        score -= len(end.group(1).strip()) * 0.001
    return score

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

# -------------------------------------------------------------------
# Model Configuration and Loading (Pure MLX)
# -------------------------------------------------------------------
# Note: MLX-LM has built-in LoRA support via command-line tools
# For training with LoRA, you can use: python -m mlx_lm.lora --model <model> --train
# This implementation focuses on full model fine-tuning with GRPO

class TiktokenTokenizerWrapper:
    """Wrapper to make tiktoken.Encoding compatible with GRPO expectations"""
    
    def __init__(self, tiktoken_tokenizer):
        self.tiktoken = tiktoken_tokenizer
        # Set up special tokens
        self.eos_token = "<|endoftext|>"
        self.pad_token = "<|endoftext|>"
        self.bos_token = "<|endoftext|>"
        
        # Get token IDs (tiktoken may not have these as properties)
        try:
            self.eos_token_id = tiktoken_tokenizer.eot_token
        except (AttributeError, KeyError):
            # If no eot_token, try encoding the token
            self.eos_token_id = tiktoken_tokenizer.encode(self.eos_token, allowed_special="all")[0]
        
        self.pad_token_id = self.eos_token_id
        self.bos_token_id = self.eos_token_id
        
        # Add properties needed by mlx_lm
        self.vocab_size = tiktoken_tokenizer.n_vocab
        self.all_special_tokens = [self.eos_token, self.pad_token, self.bos_token]
        self.all_special_ids = [self.eos_token_id, self.pad_token_id, self.bos_token_id]
        self.chat_template = None  # Nanochat doesn't have a specific chat template
        self.clean_up_tokenization_spaces = True  # Standard tokenizer behavior
        
    def encode(self, text, add_special_tokens=False):
        """Encode text to token IDs"""
        tokens = self.tiktoken.encode(text, allowed_special="all")
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens
        return tokens
    
    def decode(self, token_ids, skip_special_tokens=False):
        """Decode token IDs to text"""
        if isinstance(token_ids, list):
            return self.tiktoken.decode(token_ids)
        else:
            # Handle single token
            return self.tiktoken.decode([token_ids])
    
    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        """Apply chat template - simple version for nanochat"""
        # For nanochat, just concatenate messages with simple formatting
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted += f"{content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n"
        
        if add_generation_prompt:
            formatted += "Assistant: "
        
        if tokenize:
            return self.encode(formatted)
        return formatted
    
    def get_vocab(self):
        """Return vocab dictionary - minimal implementation for compatibility"""
        # For tiktoken, we don't have easy access to the full vocab
        # Return a minimal dict with special tokens that code might check for
        return {
            self.eos_token: self.eos_token_id,
            self.pad_token: self.pad_token_id,
            self.bos_token: self.bos_token_id,
        }

def load_tiktoken_tokenizer(model_path):
    """Load tiktoken tokenizer from pickle file"""
    model_path = Path(model_path)
    tokenizer_pkl = model_path / "tokenizer.pkl"
    
    if not tokenizer_pkl.exists():
        return None
    
    try:
        with open(tokenizer_pkl, "rb") as f:
            tiktoken_tokenizer = pickle.load(f)
        
        print(f"✅ Loaded tiktoken tokenizer (vocab size: {tiktoken_tokenizer.n_vocab})")
        return TiktokenTokenizerWrapper(tiktoken_tokenizer)
    except Exception as e:
        print(f"⚠️  Failed to load tiktoken tokenizer: {e}")
        return None

def copy_mlx_model(model):
    """Create a proper copy of an MLX model by copying all leaf arrays.

    MLX arrays are immutable; copying means creating new arrays with the same data.
    We use `mx.array(v)` to create a new array with a separate buffer.
    """
    if isinstance(model, nn.Module):
        # Get all parameters
        params = dict(model.parameters())

        # Create new arrays for all mx.array values (true copy, not shared buffer)
        new_params = {}
        for k, v in params.items():
            if isinstance(v, mx.array):
                # mx.array(existing_array) creates a new copy with separate memory
                new_params[k] = mx.array(v)
            else:
                new_params[k] = v

        # Create a new model by shallow-copying the instance and updating weights
        import copy
        new_model = copy.copy(model)  # Shallow copy (doesn't copy weights)
        new_model.update(new_params)   # Attach the new copied weights
        return new_model

    elif isinstance(model, dict) and 'model' in model:
        new_dict = {}
        for k, v in model.items():
            if k == 'model' and isinstance(v, nn.Module):
                new_dict[k] = copy_mlx_model(v)
            else:
                new_dict[k] = v
        return new_dict
    else:
        raise ValueError(f"Cannot copy model of type: {type(model)}")


def load_model(model_name):
    """Load model and tokenizer using MLX-LM."""
    import gc
    gc.collect()
    
    # Check if model_name is a local path (exists on disk)
    is_local = os.path.exists(model_name)
    
    # Try tiktoken tokenizer first (for locally converted models)
    if is_local:
        tiktoken_tok = load_tiktoken_tokenizer(model_name)
        if tiktoken_tok is not None:
            # Load model without tokenizer using mlx_lm's standard loading
            print(f"Loading model from {model_name} with tiktoken tokenizer...", flush=True)
            model, _ = mlx_load(model_name, tokenizer_config={"trust_remote_code": True})
            return model, tiktoken_tok
    
    # Standard loading with mlx_lm (works for both local and HuggingFace models)
    print(f"Loading model from {model_name}...", flush=True)
    try:
        model, tokenizer = mlx_load(model_name, tokenizer_config={"trust_remote_code": True})
        print(f"Model loaded, type: {type(model)}", flush=True)
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        raise
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer

def calculate_log_probs_single(model, tokenizer, prompt: str, completion: str) -> mx.array:
    """Return ``log p(o_i | q)`` for a single completion.

    The article computes the likelihood ratio between the trainable policy and
    the frozen rollout policy using complete reasoning traces.  To mirror that
    behaviour we feed the concatenated prompt + completion through the model
    and sum the token log probabilities for the completion span only.  The
    helper works for either ``nn.Module`` models or the dictionary-wrapped
    format returned by :mod:`mlx_lm`.
    """
    # Tokenize prompt and completion separately to know boundaries
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    # Create full sequence
    full_tokens = prompt_tokens + completion_tokens
    # Be explicit about dtype: embeddings consume integer IDs.
    input_ids = mx.array(full_tokens, dtype=mx.int32)[None, :]  # Add batch dimension

    # Forward pass through the supplied model to obtain token logits.
    if isinstance(model, nn.Module):
        logits = model(input_ids)
    elif isinstance(model, dict) and 'model' in model:
        logits = model['model'](input_ids)
    else:
        raise ValueError(f"Unexpected model type: {type(model)}")

    # Convert logits into log-probabilities over the vocabulary for every
    # timestep.  ``nn.log_softmax`` provides the numerically stable
    # implementation used throughout the GRPO derivation.
    log_probs_full = nn.log_softmax(logits, axis=-1)

    # Extract log probs for completion tokens
    # log_probs_full[i] predicts token at position i+1
    prompt_len = len(prompt_tokens)
    completion_len = len(completion_tokens)

    # Extract log-probabilities that correspond to the completion tokens only.
    completion_log_probs = []
    for i in range(completion_len):
        pos = prompt_len - 1 + i  # Position in sequence
        if pos < len(full_tokens) - 1:
            next_token_id = full_tokens[pos + 1]
            log_prob = log_probs_full[0, pos, next_token_id]
            completion_log_probs.append(log_prob)

    # Sum the completion log-probabilities to obtain log p(o_i | q).
    if len(completion_log_probs) > 0:
        return mx.sum(mx.stack(completion_log_probs))
    else:
        return mx.array(0.0)

# -------------------------------------------------------------------
# Initialize and Run GRPO Training (Pure MLX)
# -------------------------------------------------------------------
@dataclass
class MLXGRPOConfig:
    """Configuration class for MLX GRPO training"""
    # Core run metadata
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "outputs/Qwen-1.5B-MLX-GRPO"
    run_name: str = "Qwen-1.5B-MLX-GRPO-gsm8k"
    learning_rate: float = 1e-6
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_train_samples: int = 0  # Limit training samples (0 = use all)
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.1
    logging_steps: int = 1
    num_generations: int = 64  # DeepSeekMath uses 64 samples per prompt
    max_prompt_length: int = 512
    max_completion_length: int = 1024  # DeepSeekMath uses 1024
    max_new_tokens: int = 512
    temperature: float = 0.7
    clip_eps: float = 0.2  # PPO clipping epsilon
    kl_coeff: float = 0.0  # KL coefficient (can set to 0.04 as in DeepSeek)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0.0
    lr_scheduler_type: str = 'cosine'
    save_steps: int = 100
    eval_steps: int = 50  # Run EM evaluation every N steps
    eval_samples: int = 200  # Number of samples to use for evaluation
    seed: int = 0
    use_compile: bool = True  # Toggle mx.compile for gradient computation
    quantize_for_rollouts: bool = True  # Quantize model_old/ref_model to 4-bit (disable for large models)
    # --- evaluation & logging ---
    eval_every_updates: int = 25       # set 0 to disable periodic eval
    eval_subset_size: int = 200
    eval_max_new_tokens: int = 128
    log_jsonl: bool = True

# -------------------------
# Config helpers
# -------------------------
def _coerce_value(val: str, target):
    """Best-effort string -> target type coercion."""
    t = target
    if t is bool:
        return val.lower() in {"1", "true", "yes", "on"}
    if t is int:
        return int(val)
    if t is float:
        return float(val)
    return val  # str or anything else: leave as-is

def load_toml_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)

def update_config_from_dict(cfg: MLXGRPOConfig, d: Dict[str, Any]) -> MLXGRPOConfig:
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg

def apply_overrides(cfg: MLXGRPOConfig, overrides: list[str]) -> MLXGRPOConfig:
    """Override fields with --set key=value (repeatable)."""
    hints = MLXGRPOConfig.__annotations__
    for item in overrides:
        if "=" not in item:
            print(f"[warn] ignoring override (no '='): {item}")
            continue
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not hasattr(cfg, key):
            print(f"[warn] unknown config key: {key}")
            continue
        target_type = hints.get(key, str)
        try:
            coerced = _coerce_value(val, target_type)
        except Exception:
            print(f"[warn] failed to coerce '{val}' to {target_type}; using string")
            coerced = val
        setattr(cfg, key, coerced)
    return cfg

class MLXGRPOTrainer:
    def __init__(self, model, tokenizer, reward_funcs, args: MLXGRPOConfig, train_dataset, eval_dataset=None):
        self.model = model           # π_θ - trainable policy (fp16)
        self.tokenizer = tokenizer
        self.reward_funcs = reward_funcs
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Memory optimization for Apple Silicon (16GB M2 Pro):
        # - model stays in fp16 (~3GB for 1.5B) for training
        # - model_old is NOT kept in memory; we use self.model for generation
        #   and sync weights to a temp quantized copy only when needed
        # - ref_model is only kept if kl_coeff > 0
        self.model_old = None   # Lazily initialized for generation
        self._model_old_quantized = False

        if args.kl_coeff > 0:
            print("Creating ref_model (for KL penalty)...")
            self.ref_model = copy_mlx_model(model)  # π_ref
            if args.quantize_for_rollouts:
                try:
                    nn.quantize(self.ref_model, group_size=64, bits=4)
                    print("Quantized ref_model to 4-bit.")
                except Exception as e:
                    print(f"ref_model quantization skipped: {e}")
        else:
            print("kl_coeff=0: skipping ref_model to save memory.")
            self.ref_model = None

        print(f"Trainer init complete. Memory: {mx.get_active_memory() / 1e9:.2f} GB")

        # Steps / updates accounting
        self.step = 0                                    # batch steps (for logs)
        self.total_batches = len(train_dataset)
        self.updates_per_epoch = max(1, math.ceil(self.total_batches / args.gradient_accumulation_steps))
        self.total_updates = self.updates_per_epoch * args.num_epochs
        self.update_every = 10                           # Sync model_old every N *batch* steps

        # LR schedule with warmup + cosine, defined over *optimizer updates*
        base = cosine_decay(args.learning_rate, self.total_updates)
        warmup_steps = max(1, int(self.total_updates * self.args.warmup_ratio))
        def schedule(step: int):
            warm = step / warmup_steps if step < warmup_steps else 1.0
            return base(step) * warm
        self.lr_schedule = schedule
        # Use SGD to avoid Adam's ~2x memory overhead (momentum buffers)
        self.optimizer = SGD(learning_rate=self.lr_schedule)

        # Gradient accumulation
        self._accum_grads = None
        # Optimizer update counter (distinct from batch steps)
        self.update_step = 0
        # Logging path
        os.makedirs(self.args.output_dir, exist_ok=True)
        self.log_path = os.path.join(self.args.output_dir, "training_log.jsonl")

        # Tracking for logging
        self.last_reward_mean = 0.0
        self.last_reward_std = 0.0
        self.last_em_score = None

    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Build the prompt using the tokenizer's chat template (e.g., Qwen),
        falling back to the legacy string format if unavailable.
        """
        try:
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            formatted = ""
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    formatted += f"System: {content}\n\n"
                elif role == 'user':
                    formatted += f"User: {content}\n\n"
                elif role == 'assistant':
                    formatted += f"Assistant: {content}\n\n"
            formatted += "Assistant: "
            return formatted

    # -------------------------
    # JSONL logger (append)
    # -------------------------
    def _log_jsonl(self, record: Dict[str, Any]):
        if not self.args.log_jsonl:
            return
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"Logging failed: {e}")

    def generate_responses(self, batch):
        """
        Generate multiple responses for a prompt using the current model.
        Also computes old_log_probs for each response (before the update).

        Returns:
            responses: List of generated strings
            old_log_probs: Array of log probability values (mx.array)
            formatted_prompt: The formatted prompt string
        """
        messages = batch['prompt']
        formatted_prompt = self.format_prompt(messages)
        
        # Nudge the model to follow the required format
        formatted_prompt = formatted_prompt + "<reasoning>"

        responses: List[str] = []
        old_log_probs: List[float] = []  # Store as Python floats, not mx.array

        # Sampler & processors (temperature/top-p via sampler)
        sampler = make_sampler(self.args.temperature, top_p=0.95, min_p=0.0, min_tokens_to_keep=1)
        logits_processors = make_logits_processors(
            None, repetition_penalty=None, repetition_context_size=None
        )

        # Generate responses sequentially (batch gen not supported in all mlx_lm versions)
        for i in range(self.args.num_generations):
            try:
                # Generate with current model (π_θ before update)
                output = mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=self.args.max_new_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    verbose=False,
                )
                # Trim at the end tag if present to reduce junk after </answer>
                cut = output
                if "</answer>" in cut:
                    cut = cut.split("</answer>", 1)[0] + "</answer>"
                responses.append(cut)

                # Compute old_log_probs with current model (before update)
                # This is π_θ(o_i|q) - probability under the current policy
                # Materialize immediately to avoid keeping computation graph
                log_prob = calculate_log_probs_single(
                    self.model,
                    self.tokenizer,
                    formatted_prompt,
                    cut
                )
                mx.eval(log_prob)  # Force evaluation, free graph
                old_log_probs.append(float(log_prob))

            except Exception as e:
                print(f"Generation {i} failed: {e}")
                responses.append("")
                old_log_probs.append(0.0)

        # Convert to mx.array only at the end (no computation graph)
        old_log_probs_arr = mx.array(old_log_probs)

        # Clear any cached memory after generation
        mx.clear_cache()

        return responses, old_log_probs_arr, formatted_prompt
    
    def compute_rewards(self, batch, responses: List[str]) -> mx.array:
        """
        Compute rewards for all responses using reward functions.
        Returns normalized advantages based on group mean.
        """
        if len(responses) == 0:
            return mx.zeros((0,)), mx.zeros((0,))

        # Prepare completions in the format expected by reward functions.  Each
        # completion is wrapped in the structure ``[{"content": text}]`` that
        # the existing reward utilities consume.
        completions = [[{"content": response}] for response in responses]

        # Accumulate rewards from every reward function defined for the run.
        total_rewards = mx.zeros((len(responses),))
        reward_context = {
            "prompts": [batch["prompt"]],
            "answer": [batch.get("answer", "")],
        }

        for reward_fn in self.reward_funcs:
            try:
                # Inspect the callable signature to determine whether the
                # reward expects the ``prompts`` or ``answer`` keyword
                # arguments.  Functions defined in this file are permissive and
                # accept **kwargs, but the check keeps the trainer robust if a
                # custom reward omits them.
                sig = inspect.signature(reward_fn)
                kwargs: Dict[str, Any] = {}
                if "prompts" in sig.parameters:
                    kwargs["prompts"] = reward_context["prompts"]
                if "answer" in sig.parameters:
                    kwargs["answer"] = reward_context["answer"]

                reward_values = reward_fn(completions=completions, **kwargs)

                if not isinstance(reward_values, (list, tuple)):
                    reward_values = [reward_values] * len(responses)

                # Convert the return value into an array so that we can combine
                # contributions from multiple reward functions.
                reward_array = mx.array(reward_values)
                total_rewards = total_rewards + reward_array
            except Exception as e:
                print(f"Reward function failed: {e}")
                continue

        # GRPO advantage normalisation: subtract the group mean and divide by
        # the group standard deviation to obtain ``A_i``.
        mean_reward = mx.mean(total_rewards)
        std_reward = mx.std(total_rewards)
        advantages = (total_rewards - mean_reward) / (std_reward + 1e-8)

        return advantages, total_rewards

    # -------------------------
    # Quick EM evaluator
    # -------------------------
    def evaluate_em(self, dataset, num_samples: int) -> float:
        """Exact‑match on a small subset of GSM8K using the current policy."""
        if num_samples <= 0:
            return 0.0
        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        subset = idxs[:min(num_samples, len(dataset))]

        # Greedy-ish sampler (temperature 0.0)
        sampler = make_sampler(0.0, top_p=1.0, min_p=0.0, min_tokens_to_keep=1)
        logits_processors = make_logits_processors(None, repetition_penalty=None, repetition_context_size=None)

        def maybe_int(s: Optional[str]):
            if s is None:
                return None
            try:
                return int(s.strip())
            except Exception:
                return None

        correct = 0
        total = 0
        for i in subset:
            ex = dataset[i]
            gold = ex.get("answer", None)
            if gold is None:
                continue
            messages = ex["prompt"]
            try:
                prompt = self.format_prompt(messages)
                out = mlx_generate(
                    self.model, self.tokenizer,
                    prompt=prompt,
                    max_tokens=self.args.eval_max_new_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    verbose=False,
                )
                # Trim after </answer> if present
                if "</answer>" in out:
                    out = out.split("</answer>", 1)[0] + "</answer>"
                pred = extract_xml_answer(out)
                # Numeric‑aware EM
                gi, pi = maybe_int(gold), maybe_int(pred)
                if gi is not None and pi is not None:
                    match = (gi == pi)
                else:
                    match = (pred.strip() == gold.strip())
                correct += int(match)
                total += 1
            except Exception as e:
                # Skip problematic samples in eval
                continue
        return (correct / total) if total > 0 else 0.0

    def save_checkpoint(self, path: str):
        """Save model checkpoint with all necessary files for inference"""
        os.makedirs(path, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Saving checkpoint to: {path}")
        print(f"{'='*60}")
        
        # 1. Save model weights (.safetensors via Module.save_weights)
        try:
            if isinstance(self.model, nn.Module):
                self.model.save_weights(os.path.join(path, "model.safetensors"))
            elif isinstance(self.model, dict) and 'model' in self.model:
                self.model['model'].save_weights(os.path.join(path, "model.safetensors"))
            print(f"✓ Model weights saved")
        except Exception as e:
            print(f"✗ Failed to save model weights: {e}")
            return  # Critical failure - can't continue

        # 2. Copy tokenizer files (needed for inference)
        import shutil
        tokenizer_files = [
            'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
            'vocab.json', 'merges.txt', 'added_tokens.json', 'chat_template.jinja'
        ]
        
        # Determine source directory (where model was loaded from)
        source_dir = self.args.model_name
        if not os.path.isdir(source_dir):
            # Model was loaded from HF, look in cache or current dir
            if os.path.exists('tokenizer.json'):
                source_dir = '.'
        
        tokenizer_copied = 0
        for filename in tokenizer_files:
            src_path = os.path.join(source_dir, filename)
            if os.path.exists(src_path):
                try:
                    shutil.copy2(src_path, os.path.join(path, filename))
                    tokenizer_copied += 1
                except Exception:
                    pass
        
        if tokenizer_copied > 0:
            print(f"✓ Copied {tokenizer_copied} tokenizer files")
        else:
            print(f"⚠ No tokenizer files found (model may not load correctly)")

        # 3. Save model config (needed for loading)
        try:
            config_src = os.path.join(source_dir, 'config.json')
            if os.path.exists(config_src):
                shutil.copy2(config_src, os.path.join(path, 'config.json'))
                print(f"✓ Model config saved")
            else:
                print(f"⚠ config.json not found")
        except Exception as e:
            print(f"⚠ Could not copy config: {e}")

        # 4. Save optimizer state (optional - only for resuming training)
        try:
            if hasattr(self.optimizer, "state") and self.optimizer.state:
                mx.save_safetensors(os.path.join(path, "optimizer.safetensors"), self.optimizer.state)
                print(f"✓ Optimizer state saved")
        except Exception as e:
            print(f"⚠ Skipping optimizer state (known MLX issue)")

        # 5. Save training metadata
        try:
            training_state = {
                "step": int(self.step),
                "update_step": int(self.update_step),
                "epoch": 0,
                "model_name": self.args.model_name,
                "args": {k: str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v 
                        for k, v in self.args.__dict__.items()},
            }
            with open(os.path.join(path, "trainer_state.json"), "w") as f:
                json.dump(training_state, f, indent=2)
            print(f"✓ Training metadata saved")
        except Exception as e:
            print(f"⚠ Could not save training metadata: {e}")
        
        print(f"{'='*60}")
        print(f"✅ Checkpoint saved successfully!")
        print(f"{'='*60}\n")

    def compute_grpo_loss(self, policy_model, ref_model, prompt: str,
                           responses: List[str], advantages: mx.array,
                           old_log_probs: mx.array):
        """
        Compute GRPO loss following the article's implementation.

        CRITICAL: This function is called inside value_and_grad, so ALL computation
        here is recorded in the graph. We must minimize the number of forward passes.

        Args:
            policy_model: Current trainable policy (π_θ)
            ref_model: Reference model (π_ref) or None if kl_coeff=0
            prompt: The formatted prompt string
            responses: List of generated completions
            advantages: Normalized advantages (A_i)
            old_log_probs: Pre-computed log probs from model_old

        Returns:
            loss: Total loss (to be minimized)
            policy_reward_mean: Mean policy reward (for logging)
            kl_div_mean: Mean KL divergence (for logging)
        """
        if len(responses) == 0:
            zero = mx.array(0.0)
            return zero, zero, zero

        # Convert to arrays
        advantages_arr = mx.array(advantages)
        old_log_probs_arr = mx.array(old_log_probs)

        # Compute current_log_probs for ALL responses in ONE forward pass
        # by batching the inputs. This dramatically reduces graph size.
        current_log_probs = self._batch_compute_log_probs(
            policy_model, prompt, responses
        )
        current_log_probs_arr = mx.stack(current_log_probs)

        # Compute ref_log_probs if needed (KL penalty)
        if ref_model is not None:
            ref_log_probs = self._batch_compute_log_probs(
                ref_model, prompt, responses
            )
            ref_log_probs_arr = mx.stack(ref_log_probs)
        else:
            ref_log_probs_arr = mx.zeros_like(current_log_probs_arr)

        # PPO-clip objective
        ratio = mx.exp(current_log_probs_arr - old_log_probs_arr)
        clipped_ratio = mx.clip(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps)
        policy_rewards = mx.minimum(ratio * advantages_arr, clipped_ratio * advantages_arr)

        # KL penalty
        log_ratio_for_kl = ref_log_probs_arr - current_log_probs_arr
        ratio_for_kl = mx.exp(log_ratio_for_kl)
        kl_divs = ratio_for_kl - log_ratio_for_kl - 1

        policy_reward_mean = mx.mean(policy_rewards)
        kl_div_mean = mx.mean(kl_divs)

        # Total objective: maximize (policy_reward - β * KL)
        objective = policy_rewards - self.args.kl_coeff * kl_divs
        loss = -mx.mean(objective)

        return loss, policy_reward_mean, kl_div_mean

    def _batch_compute_log_probs(self, model, prompt: str, responses: List[str]):
        """
        Compute log probabilities for multiple responses using a single batched forward pass.

        Returns list of log probabilities (one per response), with gradients.
        """
        if not responses:
            return []

        # Tokenize prompt once
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        # Build batched input: list of token ID lists
        batch_tokens = []
        response_lengths = []
        max_total_len = 0

        for response in responses:
            resp_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            full_seq = prompt_tokens + resp_tokens
            batch_tokens.append(full_seq)
            response_lengths.append(len(resp_tokens))
            max_total_len = max(max_total_len, len(full_seq))

        # Pad to same length
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        padded = []
        for seq in batch_tokens:
            pad_len = max_total_len - len(seq)
            padded.append(seq + [pad_id] * pad_len)

        # Convert to batched array: shape [batch_size, max_total_len]
        input_array = mx.array(padded)

        # Single forward pass through model
        if isinstance(model, nn.Module):
            logits = model(input_array)
        elif isinstance(model, dict) and 'model' in model:
            logits = model['model'](input_array)
        else:
            raise ValueError(f"Unexpected model type: {type(model)}")

        # For each response, extract log probs of completion tokens
        # logits shape: [batch_size, seq_len, vocab_size]
        # logits[i, pos] predicts token at position pos+1
        # So for completion token at index j (0-based in response),
        # the predicting position is (prompt_len - 1 + j)
        # and the target token_id is batch_tokens[i][prompt_len + j]

        log_probs_list = []
        for i, resp_len in enumerate(response_lengths):
            if resp_len == 0:
                log_probs_list.append(mx.array(0.0))
                continue

            # Get the log probabilities for this sequence
            # We need log P(resp_tokens[j] | prompt + resp[:j])
            # = log_softmax(logits[i, prompt_len - 1 + j])[resp_tokens[j]]
            seq_log_probs = []
            for j in range(resp_len):
                # Position in the full sequence whose logits predict the next token
                pos = prompt_len - 1 + j
                if pos >= max_total_len - 1:
                    break
                # Token ID we want log prob for (the actual next token)
                # batch_tokens[i][pos + 1] is the token at position pos+1
                target_pos = pos + 1
                if target_pos >= len(batch_tokens[i]):
                    break
                token_id = batch_tokens[i][target_pos]

                # log_softmax over vocab at this position, pick our token's log prob
                log_prob = nn.log_softmax(logits[i, pos, :])[token_id]
                seq_log_probs.append(log_prob)

            if seq_log_probs:
                log_probs_list.append(mx.sum(mx.stack(seq_log_probs)))
            else:
                log_probs_list.append(mx.array(0.0))

        return log_probs_list

    def evaluate(self) -> float:
        """
        Run exact-match evaluation on a subset of test examples.
        Returns EM score (0-1).
        """
        if self.eval_dataset is None:
            return 0.0

        # Sample eval_samples examples (or use all if fewer)
        eval_size = min(self.args.eval_samples, len(self.eval_dataset))
        eval_indices = random.sample(range(len(self.eval_dataset)), eval_size)

        correct = 0
        total = 0

        # Sampler & processors for evaluation (greedy, temp=0)
        sampler = make_sampler(0.0, top_p=1.0, min_p=0.0, min_tokens_to_keep=1)
        logits_processors = make_logits_processors(
            None, repetition_penalty=None, repetition_context_size=None
        )
        # numeric-aware compare helper (same as evaluate_em)
        def maybe_int(s: Optional[str]):
            if s is None:
                return None
            try:
                return int(s.strip())
            except Exception:
                return None

        for idx in eval_indices:
            example = self.eval_dataset[idx]
            messages = example['prompt']
            gold_answer = example.get('answer', '')

            if gold_answer is None:
                continue

            formatted_prompt = self.format_prompt(messages)

            try:
                # Generate with current model (not model_old)
                output = mlx_generate(
                    self.model,
                    self.tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=self.args.eval_max_new_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    verbose=False,
                )
                # Trim after </answer> if present
                if "</answer>" in output:
                    output = output.split("</answer>", 1)[0] + "</answer>"
                predicted = extract_xml_answer(output).strip()
                gold_str = gold_answer.strip()
                # numeric-aware EM
                gi, pi = maybe_int(gold_str), maybe_int(predicted)
                match = (gi == pi) if (gi is not None and pi is not None) else (predicted == gold_str)
                correct += int(match)
                total += 1

            except Exception as e:
                print(f"Eval generation failed: {e}")
                continue

        em_score = correct / total if total > 0 else 0.0
        self.last_em_score = em_score
        print(f"\n*** Evaluation: {correct}/{total} correct (EM: {em_score:.2%}) ***")
        return em_score

    def _compute_grads(self, formatted_prompt, responses, advantages, old_log_probs):
        """Compute loss and gradients.

        Strategy: Process ONE response at a time.
        Compute gradient for each and accumulate.
        This keeps the computation graph small.
        """
        if not responses:
            return None, 0.0, 0.0, 0.0

        num_responses = len(responses)

        # Pre-compute ref_model log probs (no grad needed)
        if self.ref_model is not None:
            ref_log_probs = []
            for response in responses:
                try:
                    lp = calculate_log_probs_single(self.ref_model, self.tokenizer, formatted_prompt, response)
                    ref_log_probs.append(float(lp))
                except Exception:
                    ref_log_probs.append(0.0)
            ref_log_probs_arr = mx.array(ref_log_probs)
        else:
            ref_log_probs_arr = mx.zeros(num_responses)

        old_log_probs_arr = mx.array(old_log_probs) if isinstance(old_log_probs, list) else old_log_probs
        advantages_arr = mx.array(advantages)

        # Accumulate gradients
        accumulated_grads = None
        total_loss = 0.0
        total_policy_reward = 0.0
        total_kl_div = 0.0

        for idx in range(num_responses):
            response = responses[idx]
            resp_old_log_probs = old_log_probs_arr[idx]
            if isinstance(resp_old_log_probs, mx.array):
                resp_old_log_probs = float(resp_old_log_probs)
            resp_advantage = advantages_arr[idx]
            if isinstance(resp_advantage, mx.array):
                resp_advantage = float(resp_advantage)
            resp_ref_log_probs = ref_log_probs_arr[idx]
            if isinstance(resp_ref_log_probs, mx.array):
                resp_ref_log_probs = float(resp_ref_log_probs)

            # Build input for this single response
            prompt_tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=False)
            prompt_len = len(prompt_tokens)
            resp_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            full_seq = prompt_tokens + resp_tokens
            input_array = mx.array([full_seq])  # shape: [1, seq_len]

            # Get current params
            params = dict(self.model.trainable_parameters())

            # Define loss function that captures graph locally
            def compute_loss_for_grad(params, input_arr, prompt_len, resp_tokens, resp_old_log_prob, resp_adv, resp_ref_log_prob, full_seq):
                """Compute loss - graph is local to this function call."""
                self.model.update(params)

                # Forward pass
                if isinstance(self.model, nn.Module):
                    logits = self.model(input_arr)
                elif isinstance(self.model, dict) and 'model' in self.model:
                    logits = self.model['model'](input_arr)
                else:
                    raise ValueError(f"Unexpected model type: {type(self.model)}")

                # Compute log prob for response tokens
                resp_lps = []
                for j in range(len(resp_tokens)):
                    pos = prompt_len - 1 + j
                    if pos >= input_arr.shape[1] - 1:
                        break
                    target_pos = pos + 1
                    if target_pos >= len(full_seq):
                        break
                    target_id = full_seq[target_pos]
                    log_prob = nn.log_softmax(logits[0, pos, :])[target_id]
                    resp_lps.append(log_prob)

                if resp_lps:
                    current_log_prob = mx.sum(mx.stack(resp_lps))
                else:
                    current_log_prob = mx.array(0.0)

                # GRPO loss
                ratio = mx.exp(current_log_prob - resp_old_log_prob)
                clipped_ratio = mx.clip(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps)
                policy_reward = mx.minimum(ratio * resp_adv, clipped_ratio * resp_adv)

                if self.ref_model is not None:
                    log_ratio_for_kl = resp_ref_log_prob - current_log_prob
                    ratio_for_kl = mx.exp(log_ratio_for_kl)
                    kl_div = ratio_for_kl - log_ratio_for_kl - 1
                else:
                    kl_div = mx.array(0.0)

                objective = policy_reward - self.args.kl_coeff * kl_div
                loss = -objective
                return loss, policy_reward, kl_div

            # Compute value and grad - graph is created and destroyed within this call
            # Force stop_gradient on model outputs to prevent graph accumulation
            def compute_loss_and_grads(params):
                loss, pol_rew, kl = compute_loss_for_grad(
                    params, input_array, prompt_len, resp_tokens,
                    resp_old_log_probs, resp_advantage, resp_ref_log_probs,
                    full_seq
                )
                # Attach auxiliary values as attributes
                compute_loss_and_grads.pol_rew = pol_rew
                compute_loss_and_grads.kl = kl
                # Stop gradient to prevent graph from being retained
                return mx.stop_gradient(loss)

            # Get loss value first, then compute grads separately
            loss_val = compute_loss_and_grads(params)
            policy_reward_val = compute_loss_and_grads.pol_rew
            kl_div_val = compute_loss_and_grads.kl

            # Now compute grads with a fresh forward pass that has no retained graph
            single_grads = mx.grad(lambda p: compute_loss_and_grads(p))(params)

            # Convert to Python floats immediately to break any graph connections
            total_loss += float(loss_val)
            total_policy_reward += float(policy_reward_val)
            total_kl_div += float(kl_div_val)

            # Accumulate grads (leaf arrays, no graph)
            if accumulated_grads is None:
                accumulated_grads = single_grads
            else:
                accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, single_grads)

            # Free immediately
            del loss_val, policy_reward_val, kl_div_val, single_grads

            # Force garbage collection
            import gc
            gc.collect()
            mx.clear_cache()

        # Average metrics
        loss_val = total_loss / num_responses
        policy_reward_val = total_policy_reward / num_responses
        kl_div_val = total_kl_div / num_responses

        return accumulated_grads, loss_val, policy_reward_val, kl_div_val

    def train_step(self, batch):
        """
        Performs a single training step using GRPO.
        """
        next_step = self.step + 1
        print(f"\n{'='*60}")
        print(f"Training step {next_step}")
        print(f"{'='*60}")

        # Use a separate function to isolate computation graph
        def _do_train_step():
            nonlocal loss_val, policy_reward_val, kl_div_val
            # 1. Generate multiple responses with current model (π_θ)
            mem_before_gen = mx.get_active_memory() / 1e9
            responses, old_log_probs, formatted_prompt = self.generate_responses(batch)
            mem_after_gen = mx.get_active_memory() / 1e9
            num_responses = len(responses)
            print(f"Generated {num_responses} responses (gen memory delta: {mem_after_gen - mem_before_gen:+.2f} GB)")

            if num_responses == 0:
                print("No responses were produced; skipping update.")
                return mx.array(0.0), mx.array(0.0), mx.array(0.0)

            # 2. Compute rewards and advantages
            advantages, rewards = self.compute_rewards(batch, responses)
            self.last_reward_mean = float(mx.mean(rewards)) if num_responses > 0 else 0.0
            self.last_reward_std = float(mx.std(rewards)) if num_responses > 0 else 0.0
            adv_mean = float(mx.mean(advantages)) if num_responses > 0 else 0.0
            adv_std = float(mx.std(advantages)) if num_responses > 0 else 0.0
            print(f"Rewards - Mean: {self.last_reward_mean:.3f}, Std: {self.last_reward_std:.3f}")
            print(f"Advantages - Mean: {adv_mean:.3f}, Std: {adv_std:.3f}")

            # Display sample response
            if len(responses) > 0:
                print(f"\n--- Sample Response (Reward: {float(rewards[0]):.3f}) ---")
                print(responses[0][:200] + "..." if len(responses[0]) > 200 else responses[0])
                print(f"---")

            # 3. Compute loss and gradients
            grads, loss_val_out, policy_reward_val_out, kl_div_val_out = self._compute_grads(
                formatted_prompt, responses, advantages, old_log_probs
            )

            # Accumulate grads (leaf arrays, no graph attached)
            if self._accum_grads is None:
                self._accum_grads = grads
            else:
                self._accum_grads = tree_map(lambda a, b: a + b, self._accum_grads, grads)

            # Free per-step grads
            del grads

            do_update = ((self.step + 1) % self.args.gradient_accumulation_steps) == 0
            if do_update:
                scaled = tree_map(lambda g: g / self.args.gradient_accumulation_steps, self._accum_grads)
                scaled, grad_norm = clip_grad_norm(scaled, self.args.max_grad_norm)
                grad_norm_val = float(grad_norm)
                self.optimizer.update(self.model, scaled)
                self._accum_grads = None
                mx.clear_cache()

            loss_val = loss_val_out
            policy_reward_val = policy_reward_val_out
            kl_div_val = kl_div_val_out

            if do_update:
                print(
                    f"Loss: {loss_val:.4f}, "
                    f"GradNorm: {grad_norm_val:.4f}, "
                    f"Policy Reward: {policy_reward_val:.4f}, KL: {kl_div_val:.4f}"
                )
                self.update_step += 1
                lr_now = float(self.lr_schedule(self.update_step))
                self._log_jsonl({
                    "update": int(self.update_step),
                    "batch_step": int(self.step + 1),
                    "lr": lr_now,
                    "loss": loss_val,
                    "grad_norm": grad_norm_val,
                    "policy_reward": policy_reward_val,
                    "kl": kl_div_val,
                    "reward_mean": float(self.last_reward_mean),
                    "reward_std": float(self.last_reward_std),
                })
                if self.args.eval_every_updates > 0 and (self.update_step % self.args.eval_every_updates == 0):
                    em = self.evaluate_em(self.train_dataset, self.args.eval_subset_size)
                    print(f"[Eval] EM@{self.args.eval_subset_size}: {em:.3f}")
                    self._log_jsonl({
                        "update": int(self.update_step),
                        "em_subset": int(self.args.eval_subset_size),
                        "em": float(em),
                    })

            return mx.array(loss_val), mx.array(policy_reward_val), mx.array(kl_div_val)

        loss_val = 0.0
        policy_reward_val = 0.0
        kl_div_val = 0.0
        grad_norm_val = 0.0

        try:
            result = _do_train_step()
        except Exception as e:
            print(f"Training step failed: {e}")
            import traceback
            traceback.print_exc()
            return mx.array(0.0), mx.array(0.0), mx.array(0.0)
        finally:
            # Force cleanup
            import gc
            gc.collect()
            mx.clear_cache()

        self.step += 1
        return result

    def train(self):
        """Enhanced training loop with proper logging and checkpointing"""
        print(f"Starting training: {self.total_batches} batches/epoch × {self.args.num_epochs} epochs = {self.total_updates} optimizer updates", flush=True)
        print(f"Logging metrics to {self.log_path}", flush=True)
        print(f"Initial memory: {mx.get_active_memory() / 1e9:.2f} GB", flush=True)

        for epoch in range(self.args.num_epochs):
            indices = list(range(len(self.train_dataset)))
            random.shuffle(indices)
            
            # Limit training samples if specified
            if self.args.max_train_samples > 0:
                indices = indices[:self.args.max_train_samples]
                print(f"[INFO] Limiting training to {len(indices)} samples (max_train_samples={self.args.max_train_samples})", flush=True)

            for idx in indices:
                batch = self.train_dataset[idx]
                mem_before = mx.get_active_memory() / 1e9
                print(f"\n[Step {self.step+1}] Memory before train_step: {mem_before:.2f} GB", flush=True)
                # Each batch corresponds to one prompt/answer pair.  Iterating
                # through a shuffled epoch mimics the expectation over prompts
                # discussed in the GRPO article.
                # Training step
                loss, policy_reward, kl_div = self.train_step(batch)
                mem_after = mx.get_active_memory() / 1e9
                print(f"[Step {self.step}] Memory after train_step: {mem_after:.2f} GB (delta: {mem_after - mem_before:+.2f} GB)", flush=True)
                # Clear GPU cache to prevent memory growth
                mx.clear_cache()
                # Force garbage collection
                import gc
                gc.collect()
                # Force eval of model to break any remaining graph connections
                if isinstance(self.model, dict):
                    mx.eval([v for v in self.model.values() if isinstance(v, mx.array)])
                else:
                    mx.eval(self.model.parameters())
                mx.clear_cache()

                # (Per-update JSONL logging already handled in train_step via _log_jsonl)

                # Logging
                if self.step % self.args.logging_steps == 0:
                    current_update = self.step // self.args.gradient_accumulation_steps
                    print(
                        "Epoch {epoch}, Batch {step}/{total_batches}, Update {update}/{total_updates}, "
                        "Loss: {loss:.4f}, Policy Reward: {pr:.4f}, KL: {kl:.4f}".format(
                            epoch=epoch,
                            step=self.step,
                            total_batches=self.total_batches * self.args.num_epochs,
                            update=current_update,
                            total_updates=self.total_updates,
                            loss=float(loss),
                            pr=float(policy_reward),
                            kl=float(kl_div)
                        )
                    )
                
                # Run evaluation
                if self.step > 0 and self.step % self.args.eval_steps == 0:
                    em_score = self.evaluate()

                # Save checkpoint
                if self.step % self.args.save_steps == 0:
                    checkpoint_path = os.path.join(self.args.output_dir, f"checkpoint-{self.step}")
                    self.save_checkpoint(checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

        print(f"\nTraining log saved to {self.log_path}")

def main():
    """Main training function"""
    # ---------------- CLI / Config ----------------
    parser = argparse.ArgumentParser(description="MLX-GRPO trainer")
    parser.add_argument(
        "--config",
        default=os.environ.get("MLX_GRPO_CONFIG", "configs/default.toml"),
        help="Path to a TOML config (default: configs/default.toml or $MLX_GRPO_CONFIG).",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config keys as --set key=value (repeatable).",
    )
    args = parser.parse_args()

    # Start with defaults, then load TOML if it exists, then apply overrides
    config = MLXGRPOConfig()
    if os.path.exists(args.config):
        print(f"[config] loading {args.config}")
        toml_cfg = load_toml_config(args.config)
        # support either flat keys or a [mlx_grpo] table
        flat = toml_cfg.get("mlx_grpo", toml_cfg)
        config = update_config_from_dict(config, flat)
    else:
        print(f"[config] file not found, using defaults: {args.config}")
    if args.set:
        print(f"[config] applying overrides: {args.set}")
        config = apply_overrides(config, args.set)

    # Seed for reproducibility
    mx.random.seed(config.seed)
    random.seed(config.seed)

    # Resolve per-run output directory early to avoid clobbering
    run_dir = os.path.join(config.output_dir, config.run_name)
    os.makedirs(run_dir, exist_ok=True)
    config.output_dir = run_dir

    print("="*80)
    print("MLX-GRPO Training Pipeline")
    print("="*80)
    print(f"Model: {config.model_name}")
    print(f"Output Dir: {config.output_dir}")
    print(f"Dataset: GSM8K (train split)")
    print("="*80)

    # Persist the resolved configuration for reproducibility
    with open(os.path.join(config.output_dir, "config.resolved.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    print(f"\nTraining Configuration:")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Generations per Prompt: {config.num_generations}")
    print(f"  Max New Tokens: {config.max_new_tokens}")
    print(f"  Temperature: {config.temperature}")
    print(f"  Clip Epsilon: {config.clip_eps}")
    print(f"  KL Coefficient: {config.kl_coeff}")
    print("="*80)

    # Load dataset
    print("\nLoading dataset...")
    dataset = get_gsm8k_questions(split='train')
    eval_dataset = get_gsm8k_questions(split='test')
    print(f"Train dataset loaded: {len(dataset)} examples")
    print(f"Test dataset loaded: {len(eval_dataset)} examples")

    # Load model and tokenizer
    print("\nLoading model and tokenizer...", flush=True)
    
    # Clear any cached GPU memory before loading
    try:
        mx.metal.clear_cache()
        print("Cleared Metal cache", flush=True)
    except Exception:
        pass
    
    try:
        model, tokenizer = load_model(config.model_name)
        print(f"Model loaded successfully", flush=True)
        print(f"Model type: {type(model)}", flush=True)
        print(f"Memory after model load: {mx.get_active_memory() / 1e9:.2f} GB", flush=True)
    except Exception as e:
        print(f"Failed to load model: {e}", flush=True)
        raise

    # Force eval to ensure model is on GPU
    print("Evaluating model...", flush=True)
    mx.eval(model)
    print(f"Memory after eval: {mx.get_active_memory() / 1e9:.2f} GB", flush=True)

    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = MLXGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[
            correctness_reward_func,  # Most important - put first
            xmlcount_reward_func,
            soft_format_reward_func,
            int_reward_func,
        ],
        args=config,
        train_dataset=dataset,
        eval_dataset=eval_dataset
    )
    print("Trainer initialized")

    # Start training
    print("\n" + "="*80)
    print("Starting GRPO Training")
    print("="*80)
    trainer.train()

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Model saved to: {config.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
