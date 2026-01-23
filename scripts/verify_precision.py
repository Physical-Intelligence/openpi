#!/usr/bin/env python3
"""
Verify precision: Compare eager vs aiter attention outputs.

Ensures that optimized kernels produce numerically similar results.
"""

import os
import sys
sys.path.insert(0, "/sgl-workspace/openpi/src")
os.environ["USE_ROCM_AITER_ROPE_BACKEND"] = "0"

import numpy as np
import torch

from transformers.models.gemma.modeling_gemma import (
    set_use_aiter_attention,
    get_use_aiter_attention,
)


def create_model(device):
    """Create model."""
    original_compile = torch.compile
    torch.compile = lambda fn, **kwargs: fn
    
    import openpi.models.pi0_config as pi0_config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=10,
        max_token_len=48,
        dtype='bfloat16',
        paligemma_variant='gemma_2b',
        action_expert_variant='gemma_300m',
        pi05=False,
    )
    
    model = PI0Pytorch(config)
    model = model.to(device)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()
    
    torch.compile = original_compile
    
    return model


def create_observation(device):
    """Create deterministic observation."""
    from openpi.models import model as _model
    
    torch.manual_seed(42)
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
    
    return _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
    )


def compute_metrics(eager_out, aiter_out):
    """Compute comparison metrics."""
    eager_flat = eager_out.flatten().float()
    aiter_flat = aiter_out.flatten().float()
    
    cos_sim = torch.nn.functional.cosine_similarity(
        eager_flat.unsqueeze(0), aiter_flat.unsqueeze(0)
    ).item()
    
    abs_diff = torch.abs(eager_flat - aiter_flat)
    mean_abs_diff = abs_diff.mean().item()
    max_abs_diff = abs_diff.max().item()
    
    return {
        'cosine_similarity': cos_sim,
        'mean_abs_diff': mean_abs_diff,
        'max_abs_diff': max_abs_diff,
        'eager_has_nan': torch.isnan(eager_out).any().item(),
        'aiter_has_nan': torch.isnan(aiter_out).any().item(),
    }


def main():
    print("=" * 70)
    print("PRECISION VERIFICATION: Eager vs Aiter Attention")
    print("=" * 70)
    
    device = torch.device("cuda:0")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    
    print("\nCreating model...")
    model = create_model(device)
    
    # Force eager attention implementation (not SDPA)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
    print("Forced attention implementation: eager")
    
    # Fixed noise for both runs
    torch.manual_seed(123)
    noise = torch.randn(1, 10, 32, dtype=torch.float32, device=device)
    
    # Run with EAGER attention
    print("\n" + "-" * 50)
    print("Running with EAGER attention...")
    set_use_aiter_attention(False)
    print(f"  Aiter enabled: {get_use_aiter_attention()}")
    
    torch.manual_seed(42)
    obs = create_observation(device)
    with torch.no_grad():
        eager_actions = model.sample_actions(device, obs, noise=noise.clone(), num_steps=10)
    print(f"  Actions shape: {eager_actions.shape}")
    print(f"  Actions range: [{eager_actions.min().item():.4f}, {eager_actions.max().item():.4f}]")
    
    # Run with AITER attention
    print("\n" + "-" * 50)
    print("Running with AITER attention...")
    set_use_aiter_attention(True)
    print(f"  Aiter enabled: {get_use_aiter_attention()}")
    
    torch.manual_seed(42)
    obs = create_observation(device)
    with torch.no_grad():
        aiter_actions = model.sample_actions(device, obs, noise=noise.clone(), num_steps=10)
    print(f"  Actions shape: {aiter_actions.shape}")
    print(f"  Actions range: [{aiter_actions.min().item():.4f}, {aiter_actions.max().item():.4f}]")
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    metrics = compute_metrics(eager_actions, aiter_actions)
    
    print(f"\nCosine Similarity:     {metrics['cosine_similarity']:.6f}")
    print(f"Mean Absolute Diff:    {metrics['mean_abs_diff']:.6f}")
    print(f"Max Absolute Diff:     {metrics['max_abs_diff']:.6f}")
    print(f"Eager has NaN:         {metrics['eager_has_nan']}")
    print(f"Aiter has NaN:         {metrics['aiter_has_nan']}")
    
    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    passed = True
    
    if metrics['cosine_similarity'] > 0.99:
        print(f"[PASS] Cosine similarity {metrics['cosine_similarity']:.4f} > 0.99")
    elif metrics['cosine_similarity'] > 0.95:
        print(f"[WARN] Cosine similarity {metrics['cosine_similarity']:.4f} > 0.95 (acceptable for bf16)")
    else:
        print(f"[FAIL] Cosine similarity {metrics['cosine_similarity']:.4f} <= 0.95")
        passed = False
    
    if not metrics['eager_has_nan'] and not metrics['aiter_has_nan']:
        print("[PASS] No NaN values")
    else:
        print("[FAIL] NaN values detected")
        passed = False
    
    if metrics['max_abs_diff'] < 1.0:
        print(f"[PASS] Max absolute diff {metrics['max_abs_diff']:.4f} < 1.0")
    elif metrics['max_abs_diff'] < 2.0:
        print(f"[WARN] Max absolute diff {metrics['max_abs_diff']:.4f} < 2.0 (may be acceptable)")
    else:
        print(f"[FAIL] Max absolute diff {metrics['max_abs_diff']:.4f} >= 2.0")
        passed = False
    
    print("\n" + "=" * 70)
    if passed:
        print("RESULT: PASSED")
    else:
        print("RESULT: Differences detected (check if acceptable for bf16)")
    print("=" * 70)
    
    print("\nSample action values (first 5):")
    print(f"  Eager: {eager_actions[0, 0, :5].tolist()}")
    print(f"  Aiter: {aiter_actions[0, 0, :5].tolist()}")


if __name__ == "__main__":
    main()
