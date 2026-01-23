#!/usr/bin/env python3
"""
Benchmark full Pi0 policy inference on AMD MI350.

Uses model's built-in torch.compile on sample_actions (max-autotune mode).
"""

import os
import sys
sys.path.insert(0, "/sgl-workspace/openpi/src")

import time
import numpy as np
import torch

# Enable aiter attention
os.environ["USE_AITER_ATTENTION"] = "1"
from transformers.models.gemma.modeling_gemma import set_use_aiter_attention
set_use_aiter_attention(True)


def main():
    print("=" * 70)
    print("PI0 FULL POLICY INFERENCE BENCHMARK - AMD MI350")
    print("WITH torch.compile (max-autotune)")
    print("=" * 70)
    
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"torch.compile: ENABLED (max-autotune)")
    
    import openpi.models.pi0_config as pi0_config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    from openpi.models import model as _model
    
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=10,
        max_token_len=48,
        dtype='bfloat16',
        paligemma_variant='gemma_2b',
        action_expert_variant='gemma_300m',
        pi05=False,
    )
    
    print("\nCreating Pi0 model with torch.compile...")
    model = PI0Pytorch(config)
    model = model.to(device)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model parameters: {param_count:.2f}B")
    
    # Create observation
    batch_size = 1
    images = {
        'base_0_rgb': torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
        'left_wrist_0_rgb': torch.rand(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
        'right_wrist_0_rgb': torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32, device=device),
    }
    image_masks = {
        'base_0_rgb': torch.ones(batch_size, dtype=torch.bool, device=device),
        'left_wrist_0_rgb': torch.ones(batch_size, dtype=torch.bool, device=device),
        'right_wrist_0_rgb': torch.zeros(batch_size, dtype=torch.bool, device=device),
    }
    state = torch.randn(batch_size, 32, dtype=torch.bfloat16, device=device)
    tokenized_prompt = torch.randint(0, 256000, (batch_size, 20), dtype=torch.long, device=device)
    tokenized_prompt_mask = torch.ones(batch_size, 20, dtype=torch.bool, device=device)
    
    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )
    
    num_steps = 10
    warmup = 5
    iterations = 20
    
    # Warmup (includes torch.compile compilation)
    print("\nWarmup (includes torch.compile compilation)...")
    for i in range(warmup):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.sample_actions(device, observation, num_steps=num_steps)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Warmup {i+1}: {elapsed:.1f} ms")
    
    # Benchmark
    print("Benchmarking...")
    latencies = []
    for i in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            actions = model.sample_actions(device, observation, num_steps=num_steps)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)
    print(f"Mean latency:   {np.mean(latencies):.1f} ms")
    print(f"Std:            {np.std(latencies):.1f} ms")
    print(f"Throughput:     {1000/np.mean(latencies):.2f} Hz")
    print(f"Actions shape:  {tuple(actions.shape)}")
    print(f"Memory:         {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
    print("=" * 70)


if __name__ == "__main__":
    main()
