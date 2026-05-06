import json
import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.tuner import TrainingArgs, datasets, linear_to_lora_layers, train


model_path = "mlx-community/SmolLM-135M-4bit"
model, tokenizer = load(model_path)

prompt = "What is fine-tuning in machine learning?"
messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
response = generate(model, tokenizer, prompt=prompt, verbose=True)


adapter_path = "adapters"
os.makedirs(adapter_path, exist_ok=True)
adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
adapter_file_path = os.path.join(adapter_path, "adapters.safetensors")

lora_config = {
    "num_layers": 8,
    "lora_parameters": {
        "rank": 8,
        "scale": 20.0,
        "dropout": 0.0,
    },
}

with open(adapter_config_path, "w") as f:
    json.dump(lora_config, f, indent=4)

training_args = TrainingArgs(
    adapter_file=adapter_file_path,
    iters=200,
    steps_per_eval=50,
)
#loss = cross entropy, when using Low-Rank Adaptation (LoRA) for Supervised Fine-Tuning (SFT), a reward function is typically not required.

model.freeze()
linear_to_lora_layers(model, lora_config["num_layers"], lora_config["lora_parameters"]) 
num_train_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
print(f"Number of trainable parameters: {num_train_params}")

model.train()

class Metrics:
    def __init__(self) -> None:
        self.train_losses: List[Tuple[int, float]] = []
        self.val_losses: List[Tuple[int, float]] = []

    def on_train_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.train_losses.append((info["iteration"], info["train_loss"]))

    def on_val_loss_report(self, info: Dict[str, Union[float, int]]) -> None:
        self.val_losses.append((info["iteration"], info["val_loss"]))

metrics = Metrics()
def custom_load_hf_dataset(
    data_id: str,
    tokenizer,
    names: Tuple[str, str, str] = ("train", "valid", "test"),
):
    from datasets import exceptions, load_dataset

    try:
        dataset = load_dataset(data_id)

        train, valid, test = [
            dataset[n] if n in dataset.keys() else None
            for n in names
        ]

    except exceptions.DatasetNotFoundError:
        raise ValueError(f"Not found Hugging Face dataset: {data_id} .")

    return train, valid, test


def format_dataset_for_training(dataset, tokenizer, training_args):
    """Format dataset into MLX-LM training format."""
    formatted_data = []

    # Check the actual format of the first example to understand the structure
    if len(dataset) > 0:
        first_example = dataset[0]
        print(f"\nDataset example keys: {first_example.keys()}")
        print(f"First example: {first_example}\n")

    for example in dataset:
        # The dataset appears to use 'text' field with Gemma format
        if 'text' in example:
            text = example['text']
        elif 'question' in example and 'answer' in example:
            # Fallback to QA format if that's the case
            text = f"<start_of_turn>user\n{example['question']}<end_of_turn>\n<start_of_turn>model\n{example['answer']}<end_of_turn>"
        else:
            continue

        formatted_data.append({"text": text})

    # Use mask_prompt=False to treat it as standard text completion
    class SimpleTextDataset:
        def __init__(self, data, tokenizer):
            self.data = data
            self.tokenizer = tokenizer

        def __getitem__(self, idx):
            text = self.data[idx]["text"]
            tokens = self.tokenizer.encode(text)
            if tokens[-1] != self.tokenizer.eos_token_id:
                tokens.append(self.tokenizer.eos_token_id)
            return (tokens, 0)

        def __len__(self):
            return len(self.data)

    return SimpleTextDataset(formatted_data, tokenizer)


raw_train, raw_val, raw_test = custom_load_hf_dataset(
    data_id="win-wang/Machine_Learning_QA_Collection",
    tokenizer=tokenizer,
    names=("train", "validation", "test"),
)

train_set = format_dataset_for_training(raw_train, tokenizer, training_args) if raw_train else None
val_set = format_dataset_for_training(raw_val, tokenizer, training_args) if raw_val else None
test_set = format_dataset_for_training(raw_test, tokenizer, training_args) if raw_test else None

# Print first 5 parsed instances of the training dataset
if train_set:
    print("\n" + "="*80)
    print("First 5 parsed instances of the training dataset:")
    print("="*80)
    for i in range(min(5, len(train_set))):
        tokens, offset = train_set[i]
        decoded_text = tokenizer.decode(tokens)
        print(f"\nInstance {i+1}:")
        print(f"  Tokens length: {len(tokens)}")
        print(f"  Offset: {offset}")
        print(f"  Decoded text (first 200 chars): {decoded_text[:200]}...")
        print(f"  Full tokens: {tokens[:20]}..." if len(tokens) > 20 else f"  Full tokens: {tokens}")
    print("="*80 + "\n")

train(
    model=model,
    #tokenizer=tokenizer,
    args=training_args,
    optimizer=optim.Adam(learning_rate=1e-5),
    train_dataset=train_set,
    val_dataset=val_set,
    training_callback=metrics,
)

train_its, train_losses = zip(*metrics.train_losses)
validation_its, validation_losses = zip(*metrics.val_losses)
plt.plot(train_its, train_losses, "-o", label="Train")
plt.plot(validation_its, validation_losses, "-o", label="Validation")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.show()


############


#model_lora, _ = load(model_path, adapter_path=adapter_path)
#response = generate(model_lora, tokenizer, prompt=prompt, verbose=True)

#try https://github.com/ARahim3/mlx-tune