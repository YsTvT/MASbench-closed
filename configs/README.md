# Real Minecraft GRPO configuration

`verl_minecraft_grpo_lora.yaml` is the canonical veRL/verl-agent pilot config.
It uses shared Qwen2.5-7B LoRA (rank 32, all-linear), stochastic group
sampling (`n=4`), GRPO group-relative advantages, action-token KL and a real
Minecraft checker. Every candidate must reset the same task-34 world snapshot.

This file is a configuration contract; it does not make the older
`train_real_rl.py` runner a veRL trainer. The pilot is valid only after the
Minecraft environment package is registered in `verl-agent` and the preflight
checks pass.
