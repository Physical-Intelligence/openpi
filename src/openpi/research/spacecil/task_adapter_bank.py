"""Per-task PEFT module registry using openpi's LoRA infrastructure.

Each task adapter = separate LoRA parameter set (nnx.State filtered to .*lora.*).
Bank = dictionary {task_id: lora_state_dict}.
Swap via nnx.update(model, lora_state) before forward passes.
"""
