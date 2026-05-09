# Training LLMs on MLX

MLX LLM training scripts for finetuning that really work on a Mac Book Pro. 

- Vanilla LORA: Idea from [Joana Levtcheva's post](https://medium.com/@levchevajoana/fine-tuning-llms-with-lora-and-mlx-lm-c0b143642deb). Fixed the non-working part (custom_load_hf_dataset) using OpenCode and replaced `Mistral-7B-Instruct-v0.3-4bit` with `SmolLM-135M-4bit`. Run `python mlx_finetune.py` in a `venv` environment.

- GRPO: Idea from [Doriandarko's MLX-GRPO](https://github.com/Doriandarko/MLX-GRPO). Fit the finetuning pipeline into M2 Pro 16GB using OpenCode by applying `mx.stop_gradient()` on the loss value in `_compute_grads()` to break computation graph retention. Runtime memory use is between 3GB - 7GB. Run `uv run mlx-grpo.py --config configs/default.toml` in a `venv` environment.

# Notes

- A nice step-by-step article about [GRPO](https://github.com/searlion/mlx-finetuning/blob/main/MLX%20LM%20GRPO.ipynb).

# TODO 

- SFT, DPO, GRPO, TTS, STT using [mlx-tune](https://github.com/ARahim3/mlx-tune) 
- Tool calls

