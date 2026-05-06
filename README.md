# mlx_llm_training

Working mlx llm training script for finetuning. Idea from [Joana Levtcheva's post](https://medium.com/@levchevajoana/fine-tuning-llms-with-lora-and-mlx-lm-c0b143642deb).

Fixed the non-working part (custom_load_hf_dataset) using OpenCode and replaced `Mistral-7B-Instruct-v0.3-4bit` with `SmolLM-135M-4bit`. 

# TODO 

- SFT, DPO, GRPO, TTS, STT using [mlx-tune](https://github.com/ARahim3/mlx-tune)
- [GRPO](https://github.com/searlion/mlx-finetuning/blob/main/MLX%20LM%20GRPO.ipynb), [GRPO2](https://github.com/Doriandarko/MLX-GRPO)
- Tool calls
