#!/usr/bin/env python3
"""
AMD MI350 Benchmark for OpenPI - Optimized Kernels (Aiter Flash Attention + Triton)

Covers inference and training across various workload configurations.
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
from torch.profiler import profile, ProfilerActivity

from transformers.models.gemma.modeling_gemma import (
    set_use_aiter_attention,
    get_use_aiter_attention,
    GemmaAttention,
)
from transformers.models.gemma.configuration_gemma import GemmaConfig

from openpi.models_pytorch.triton_ops import (
    rms_norm_triton,
    gelu_tanh_and_mul_triton,
)


class GemmaLayerOptimized(nn.Module):
    """Single Gemma layer with optimized kernels (Aiter + Triton)."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # Attention
        self.attention = GemmaAttention(config, layer_idx=0)
        
        # MLP
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        
        # Norms
        self.input_norm_weight = nn.Parameter(torch.ones(config.hidden_size))
        self.post_attn_norm_weight = nn.Parameter(torch.ones(config.hidden_size))
    
    def forward(self, x, position_embeddings, attention_mask):
        # Pre-attention norm (Triton RMSNorm)
        normed = rms_norm_triton(x, self.input_norm_weight)
        
        # Attention (Aiter Flash Attention)
        attn_out, _ = self.attention(
            hidden_states=normed,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden = x + attn_out
        
        # Post-attention norm (Triton RMSNorm)
        normed = rms_norm_triton(hidden, self.post_attn_norm_weight)
        
        # MLP with fused GELU (Triton)
        gate_up = self.gate_up_proj(normed)
        mlp_out = gelu_tanh_and_mul_triton(gate_up)
        mlp_out = self.down_proj(mlp_out)
        
        return hidden + mlp_out


class MultiLayerModel(nn.Module):
    """Multi-layer model for realistic benchmarking."""
    
    def __init__(self, config, num_layers=4):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([GemmaLayerOptimized(config) for _ in range(num_layers)])
        self.final_norm = nn.Parameter(torch.ones(config.hidden_size))
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, x, position_embeddings, attention_mask, labels=None):
        for layer in self.layers:
            x = layer(x, position_embeddings, attention_mask)
        
        x = rms_norm_triton(x, self.final_norm)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )
        
        return logits, loss


def create_inputs(device, batch_size, seq_len, config, for_training=False):
    """Create model inputs."""
    dtype = torch.bfloat16
    
    x = torch.randn(batch_size, seq_len, config.hidden_size, dtype=dtype, device=device)
    cos = torch.randn(batch_size, seq_len, config.head_dim, dtype=dtype, device=device)
    sin = torch.randn(batch_size, seq_len, config.head_dim, dtype=dtype, device=device)
    
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
        diagonal=1
    )
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    
    labels = None
    if for_training:
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    
    return x, (cos, sin), attention_mask, labels


def benchmark_inference(model, inputs, warmup=20, iterations=100):
    """Benchmark inference throughput."""
    x, pos_emb, mask, _ = inputs
    batch_size = x.shape[0]
    
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, pos_emb, mask)
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x, pos_emb, mask)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed = end - start
    total_samples = batch_size * iterations
    samples_per_sec = total_samples / elapsed
    ms_per_sample = (elapsed / total_samples) * 1000
    
    return {
        'samples_per_sec': samples_per_sec,
        'ms_per_sample': ms_per_sample,
        'total_time_s': elapsed,
    }


def benchmark_training(model, inputs, warmup=10, iterations=50):
    """Benchmark training throughput."""
    x, pos_emb, mask, labels = inputs
    batch_size = x.shape[0]
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Warmup
    for _ in range(warmup):
        optimizer.zero_grad()
        _, loss = model(x, pos_emb, mask, labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad()
        _, loss = model(x, pos_emb, mask, labels)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed = end - start
    total_samples = batch_size * iterations
    samples_per_sec = total_samples / elapsed
    ms_per_step = (elapsed / iterations) * 1000
    
    return {
        'samples_per_sec': samples_per_sec,
        'ms_per_step': ms_per_step,
        'total_time_s': elapsed,
        'final_loss': loss.item(),
    }


def profile_and_save(model, inputs, name, output_dir, is_training=False):
    """Profile model and save trace."""
    x, pos_emb, mask, labels = inputs
    
    os.makedirs(output_dir, exist_ok=True)
    trace_path = os.path.join(output_dir, f"{name}.json")
    
    # Warmup
    if is_training:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(3):
            optimizer.zero_grad()
            _, loss = model(x, pos_emb, mask, labels)
            loss.backward()
            optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                _ = model(x, pos_emb, mask)
    torch.cuda.synchronize()
    
    # Profile with more iterations for better trace
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        if is_training:
            for _ in range(5):
                optimizer.zero_grad()
                _, loss = model(x, pos_emb, mask, labels)
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x, pos_emb, mask)
        torch.cuda.synchronize()
    
    # Export
    prof.export_chrome_trace(trace_path)
    
    # Get size
    size_mb = os.path.getsize(trace_path) / 1024 / 1024
    
    return trace_path, size_mb


def main():
    print("=" * 80)
    print("AMD MI350 BENCHMARK - OpenPI Optimized Kernels")
    print("Aiter Flash Attention + Triton Elementwise Ops")
    print("=" * 80)
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\nDevice: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    
    # Enable optimized kernels
    set_use_aiter_attention(True)
    
    # Model configurations - use head_dim=256 which works with aiter
    model_configs = [
        {"name": "Small", "hidden": 1024, "heads": 4, "head_dim": 256, "intermediate": 4096, "layers": 4, "vocab": 32000},
        {"name": "Medium", "hidden": 2048, "heads": 8, "head_dim": 256, "intermediate": 8192, "layers": 8, "vocab": 32000},
        {"name": "Large", "hidden": 4096, "heads": 16, "head_dim": 256, "intermediate": 16384, "layers": 12, "vocab": 32000},
    ]
    
    # Workload configurations
    workloads = [
        {"batch": 1, "seq": 512, "desc": "Single sample, medium seq"},
        {"batch": 4, "seq": 512, "desc": "Small batch, medium seq"},
        {"batch": 8, "seq": 1024, "desc": "Medium batch, long seq"},
        {"batch": 16, "seq": 512, "desc": "Large batch, medium seq"},
        {"batch": 4, "seq": 2048, "desc": "Small batch, very long seq"},
        {"batch": 32, "seq": 256, "desc": "Very large batch, short seq"},
    ]
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "gpu": gpu_name,
            "pytorch": torch.__version__,
            "optimizations": "Aiter Flash Attention + Triton RMSNorm/GELU",
        },
        "inference": [],
        "training": [],
        "traces": [],
    }
    
    trace_dir = "/sgl-workspace/openpi/traces"
    os.makedirs(trace_dir, exist_ok=True)
    
    # Run benchmarks for each model config
    for mcfg in model_configs:
        print(f"\n{'='*80}")
        print(f"MODEL: {mcfg['name']} (hidden={mcfg['hidden']}, layers={mcfg['layers']})")
        print("=" * 80)
        
        config = GemmaConfig(
            hidden_size=mcfg['hidden'],
            num_attention_heads=mcfg['heads'],
            num_key_value_heads=mcfg['heads'],
            head_dim=mcfg['head_dim'],
            intermediate_size=mcfg['intermediate'],
            num_hidden_layers=mcfg['layers'],
            vocab_size=mcfg['vocab'],
            attention_dropout=0.0,
            attention_bias=False,
        )
        config._attn_implementation = "eager"
        
        model = MultiLayerModel(config, num_layers=mcfg['layers']).to(device).to(torch.bfloat16)
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Parameters: {param_count:.1f}M")
        
        # Test each workload
        for wl in workloads:
            batch, seq = wl['batch'], wl['seq']
            
            # Check memory
            try:
                torch.cuda.empty_cache()
                inputs = create_inputs(device, batch, seq, config, for_training=True)
            except RuntimeError as e:
                print(f"  B{batch}xS{seq}: OOM - skipped")
                continue
            
            print(f"\n  Workload: B{batch}xS{seq} ({wl['desc']})")
            print(f"  {'-'*60}")
            
            # Inference benchmark
            try:
                inf_result = benchmark_inference(model, inputs, warmup=10, iterations=50)
                print(f"  Inference: {inf_result['samples_per_sec']:>10.1f} samples/s | {inf_result['ms_per_sample']:.3f} ms/sample")
                
                results['inference'].append({
                    'model': mcfg['name'],
                    'batch': batch,
                    'seq': seq,
                    **inf_result
                })
            except Exception as e:
                print(f"  Inference: ERROR - {str(e)[:60]}")
            
            # Training benchmark
            try:
                # Recreate model for clean training state
                model = MultiLayerModel(config, num_layers=mcfg['layers']).to(device).to(torch.bfloat16)
                train_result = benchmark_training(model, inputs, warmup=5, iterations=20)
                print(f"  Training:  {train_result['samples_per_sec']:>10.1f} samples/s | {train_result['ms_per_step']:.1f} ms/step | loss={train_result['final_loss']:.4f}")
                
                results['training'].append({
                    'model': mcfg['name'],
                    'batch': batch,
                    'seq': seq,
                    **train_result
                })
            except Exception as e:
                print(f"  Training: ERROR - {str(e)[:60]}")
            
            torch.cuda.empty_cache()
        
        # Generate traces for representative workload
        print(f"\n  Generating traces...")
        try:
            model = MultiLayerModel(config, num_layers=mcfg['layers']).to(device).to(torch.bfloat16)
            inputs = create_inputs(device, 8, 1024, config, for_training=True)
            
            # Inference trace
            trace_path, size_mb = profile_and_save(
                model, inputs, 
                f"{mcfg['name'].lower()}_inference", 
                trace_dir, 
                is_training=False
            )
            print(f"  Inference trace: {size_mb:.1f} MB")
            results['traces'].append({'name': f"{mcfg['name']}_inference", 'path': trace_path, 'size_mb': size_mb})
            
            # Training trace
            model = MultiLayerModel(config, num_layers=mcfg['layers']).to(device).to(torch.bfloat16)
            trace_path, size_mb = profile_and_save(
                model, inputs, 
                f"{mcfg['name'].lower()}_training", 
                trace_dir, 
                is_training=True
            )
            print(f"  Training trace:  {size_mb:.1f} MB")
            results['traces'].append({'name': f"{mcfg['name']}_training", 'path': trace_path, 'size_mb': size_mb})
            
        except Exception as e:
            print(f"  Trace generation failed: {str(e)[:60]}")
        
        del model
        torch.cuda.empty_cache()
    
    # Save results
    results_path = "/sgl-workspace/openpi/benchmark_results_mi350.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY - BEST RESULTS")
    print("=" * 80)
    
    if results['inference']:
        best_inf = max(results['inference'], key=lambda x: x['samples_per_sec'])
        print(f"\nBest Inference: {best_inf['samples_per_sec']:.1f} samples/s")
        print(f"  Model: {best_inf['model']}, Batch: {best_inf['batch']}, Seq: {best_inf['seq']}")
    
    if results['training']:
        best_train = max(results['training'], key=lambda x: x['samples_per_sec'])
        print(f"\nBest Training: {best_train['samples_per_sec']:.1f} samples/s")
        print(f"  Model: {best_train['model']}, Batch: {best_train['batch']}, Seq: {best_train['seq']}")
    
    print("\n" + "=" * 80)
    print("INFERENCE RESULTS")
    print("=" * 80)
    print(f"{'Model':<8} {'Batch':<6} {'Seq':<6} {'Samples/s':<12} {'ms/sample':<10}")
    print("-" * 50)
    for r in sorted(results['inference'], key=lambda x: (x['model'], x['batch'], x['seq'])):
        print(f"{r['model']:<8} {r['batch']:<6} {r['seq']:<6} {r['samples_per_sec']:<12.1f} {r['ms_per_sample']:<10.3f}")
    
    print("\n" + "=" * 80)
    print("TRAINING RESULTS")
    print("=" * 80)
    print(f"{'Model':<8} {'Batch':<6} {'Seq':<6} {'Samples/s':<12} {'ms/step':<10}")
    print("-" * 50)
    for r in sorted(results['training'], key=lambda x: (x['model'], x['batch'], x['seq'])):
        print(f"{r['model']:<8} {r['batch']:<6} {r['seq']:<6} {r['samples_per_sec']:<12.1f} {r['ms_per_step']:<10.1f}")
    
    print("\n" + "=" * 80)
    print("TRACES")
    print("=" * 80)
    for t in results['traces']:
        print(f"  {t['name']}: {t['size_mb']:.1f} MB")
    
    print("\nView traces at: https://ui.perfetto.dev/")
    print("=" * 80)


if __name__ == "__main__":
    main()
