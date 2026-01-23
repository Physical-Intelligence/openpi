#!/usr/bin/env python3
"""
AMD MI350 Multi-GPU DDP Benchmark - Large Scale Training

Uses all 8 GPUs with overlapping computation and communication.
"""

import os
import sys
sys.path.insert(0, "/sgl-workspace/openpi/src")

import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity

from transformers.models.gemma.modeling_gemma import (
    set_use_aiter_attention,
    GemmaAttention,
)
from transformers.models.gemma.configuration_gemma import GemmaConfig

# Note: Using standard PyTorch ops for DDP compatibility
# Aiter flash attention is still used via GemmaAttention


class GemmaLayerOptimized(nn.Module):
    """Single Gemma layer with Aiter flash attention."""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.attention = GemmaAttention(config, layer_idx=0)
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # Use nn.RMSNorm for DDP compatibility
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-6)
    
    def forward(self, x, position_embeddings, attention_mask):
        # Pre-attention norm
        normed = self.input_layernorm(x)
        # Attention (Aiter Flash Attention)
        attn_out, _ = self.attention(hidden_states=normed, position_embeddings=position_embeddings, attention_mask=attention_mask)
        hidden = x + attn_out
        # Post-attention norm
        normed = self.post_attention_layernorm(hidden)
        # MLP
        gate_up = self.gate_up_proj(normed)
        gate = gate_up[..., :self.intermediate_size]
        up = gate_up[..., self.intermediate_size:]
        mlp_out = F.gelu(gate, approximate='tanh') * up
        mlp_out = self.down_proj(mlp_out)
        return hidden + mlp_out


class MultiLayerModel(nn.Module):
    """Multi-layer model for DDP training."""
    
    def __init__(self, config, num_layers=12):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([GemmaLayerOptimized(config) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, x, position_embeddings, attention_mask, labels=None):
        for layer in self.layers:
            x = layer(x, position_embeddings, attention_mask)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return logits, loss


def create_inputs(device, batch_size, seq_len, config):
    """Create training inputs."""
    dtype = torch.bfloat16
    x = torch.randn(batch_size, seq_len, config.hidden_size, dtype=dtype, device=device)
    cos = torch.randn(batch_size, seq_len, config.head_dim, dtype=dtype, device=device)
    sin = torch.randn(batch_size, seq_len, config.head_dim, dtype=dtype, device=device)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), diagonal=1)
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    return x, (cos, sin), attention_mask, labels


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def benchmark_ddp_training(rank, world_size, config, batch_per_gpu, seq_len, num_layers, warmup=5, iterations=20):
    """Benchmark DDP training with overlapping."""
    device = torch.device(f'cuda:{rank}')
    
    # Create model and wrap with DDP
    model = MultiLayerModel(config, num_layers=num_layers).to(device).to(torch.bfloat16)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create inputs
    inputs = create_inputs(device, batch_per_gpu, seq_len, config)
    x, pos_emb, mask, labels = inputs
    
    # Warmup
    model.train()
    for _ in range(warmup):
        optimizer.zero_grad()
        _, loss = model(x, pos_emb, mask, labels)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    dist.barrier()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad()
        _, loss = model(x, pos_emb, mask, labels)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    dist.barrier()
    end = time.perf_counter()
    
    elapsed = end - start
    total_samples = batch_per_gpu * world_size * iterations
    samples_per_sec = total_samples / elapsed
    ms_per_step = (elapsed / iterations) * 1000
    
    return {
        'samples_per_sec': samples_per_sec,
        'ms_per_step': ms_per_step,
        'total_time_s': elapsed,
        'final_loss': loss.item(),
        'world_size': world_size,
        'batch_per_gpu': batch_per_gpu,
        'effective_batch': batch_per_gpu * world_size,
    }


def profile_ddp_training(rank, world_size, config, batch_per_gpu, seq_len, num_layers, output_dir):
    """Profile DDP training and save trace."""
    device = torch.device(f'cuda:{rank}')
    
    model = MultiLayerModel(config, num_layers=num_layers).to(device).to(torch.bfloat16)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    inputs = create_inputs(device, batch_per_gpu, seq_len, config)
    x, pos_emb, mask, labels = inputs
    
    # Warmup
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        _, loss = model(x, pos_emb, mask, labels)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    dist.barrier()
    
    # Profile
    trace_path = os.path.join(output_dir, f"ddp_training_rank{rank}.json")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for _ in range(5):
            optimizer.zero_grad()
            _, loss = model(x, pos_emb, mask, labels)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
    
    prof.export_chrome_trace(trace_path)
    
    return trace_path


def main():
    # Setup distributed
    local_rank, world_size = setup_distributed()
    
    is_main = (local_rank == 0)
    
    if is_main:
        print("=" * 80)
        print(f"AMD MI350 DDP BENCHMARK - {world_size} GPUs")
        print("Aiter Flash Attention + Triton Ops")
        print("=" * 80)
        print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch: {torch.__version__}")
    
    # Enable optimized kernels
    set_use_aiter_attention(True)
    
    # Model config - Large model for multi-GPU
    config = GemmaConfig(
        hidden_size=4096,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        intermediate_size=16384,
        num_hidden_layers=12,
        vocab_size=32000,
        attention_dropout=0.0,
        attention_bias=False,
    )
    config._attn_implementation = "eager"
    
    num_layers = 12
    param_count = 3352.4  # M
    
    if is_main:
        print(f"\nModel: Large ({param_count:.1f}M params, {num_layers} layers)")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "gpu": torch.cuda.get_device_name(0),
            "num_gpus": world_size,
            "pytorch": torch.__version__,
        },
        "ddp_training": [],
        "traces": [],
    }
    
    # Test configurations
    test_configs = [
        {"batch_per_gpu": 4, "seq": 512, "desc": "Small batch"},
        {"batch_per_gpu": 8, "seq": 512, "desc": "Medium batch"},
        {"batch_per_gpu": 8, "seq": 1024, "desc": "Long sequence"},
        {"batch_per_gpu": 16, "seq": 512, "desc": "Large batch"},
    ]
    
    for tc in test_configs:
        batch_per_gpu = tc['batch_per_gpu']
        seq_len = tc['seq']
        effective_batch = batch_per_gpu * world_size
        
        if is_main:
            print(f"\n{'='*70}")
            print(f"Config: B{batch_per_gpu}/GPU x {world_size}GPU = B{effective_batch} total, S{seq_len}")
            print(f"({tc['desc']})")
            print("-" * 70)
        
        try:
            result = benchmark_ddp_training(
                local_rank, world_size, config,
                batch_per_gpu, seq_len, num_layers,
                warmup=5, iterations=20
            )
            
            if is_main:
                print(f"Throughput: {result['samples_per_sec']:.1f} samples/s")
                print(f"Step time:  {result['ms_per_step']:.1f} ms")
                print(f"Loss:       {result['final_loss']:.4f}")
                
                results['ddp_training'].append({
                    'batch_per_gpu': batch_per_gpu,
                    'seq': seq_len,
                    'effective_batch': effective_batch,
                    **result
                })
        except Exception as e:
            if is_main:
                print(f"ERROR: {str(e)[:60]}")
        
        torch.cuda.empty_cache()
        dist.barrier()
    
    # Generate traces
    trace_dir = "/sgl-workspace/openpi/traces"
    os.makedirs(trace_dir, exist_ok=True)
    
    if is_main:
        print(f"\n{'='*70}")
        print("Generating DDP traces...")
    
    try:
        trace_path = profile_ddp_training(
            local_rank, world_size, config,
            batch_per_gpu=8, seq_len=1024, num_layers=num_layers,
            output_dir=trace_dir
        )
        
        dist.barrier()
        
        if is_main:
            # Check all trace files
            for r in range(world_size):
                tp = os.path.join(trace_dir, f"ddp_training_rank{r}.json")
                if os.path.exists(tp):
                    size = os.path.getsize(tp) / 1024 / 1024
                    print(f"  Rank {r}: {size:.1f} MB")
                    results['traces'].append({'rank': r, 'path': tp, 'size_mb': size})
    except Exception as e:
        if is_main:
            print(f"Trace generation failed: {str(e)[:60]}")
    
    # Save results (only main rank)
    if is_main:
        results_path = "/sgl-workspace/openpi/benchmark_results_mi350_ddp.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
        
        # Summary
        print("\n" + "=" * 80)
        print("DDP TRAINING SUMMARY")
        print("=" * 80)
        print(f"{'Batch/GPU':<12} {'Total Batch':<12} {'Seq':<6} {'Samples/s':<12} {'ms/step':<10}")
        print("-" * 55)
        for r in results['ddp_training']:
            print(f"{r['batch_per_gpu']:<12} {r['effective_batch']:<12} {r['seq']:<6} {r['samples_per_sec']:<12.1f} {r['ms_per_step']:<10.1f}")
        
        if results['ddp_training']:
            best = max(results['ddp_training'], key=lambda x: x['samples_per_sec'])
            print("-" * 55)
            print(f"\nBest: {best['samples_per_sec']:.1f} samples/s @ B{best['effective_batch']}xS{best['seq']}")
        
        print("\nTraces for Perfetto: https://ui.perfetto.dev/")
        print("=" * 80)
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
