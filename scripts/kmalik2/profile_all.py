#!/usr/bin/env python3
"""
Profiling script for FeedForward inference with different configurations.
Usage: python profile_all.py [--dryrun] [--include-no-jit] [--batch-size BATCH_SIZE] [--num-shards NUM_SHARDS]

CUDA devices are automatically configured based on --num-shards parameter.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Profile FeedForward inference with different configurations")
    parser.add_argument("--dryrun", action="store_true", help="Print commands without executing them")
    parser.add_argument("--include-no-jit", action="store_true", help="Also run non-JIT versions (default: JIT only)")
    parser.add_argument("--batch-size", type=int, help="Batch size (default: same as num_shards)")
    parser.add_argument("--num-shards", type=int, default=8, help="Number of GPU shards to use (default: 2)")
    args = parser.parse_args()
    
    # Set batch size to same as num_shards if not provided
    num_shards = args.num_shards
    batch_size = args.batch_size if args.batch_size is not None else num_shards
    
    print("=== FeedForward Profiling Script ===")
    print(f"Dry run mode: {args.dryrun}")
    print(f"Include no-JIT versions: {args.include_no_jit}")
    print(f"Number of shards: {num_shards}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Base command configuration
    base_cmd = [
        "uv", "run", "scripts/kmalik2/test_feedforward_inference.py",
        "--mode", "profile",
        "--num-shards", str(num_shards),
        "--batch-size", str(batch_size),
        "--seq-len", "2048"
    ]
    
    # Configuration combinations - only sharded versions
    configs = [
        {
            "name": "hidden_larger",
            "model_dim": 2048,
            "hidden_dim": 8192
        },
        {
            "name": "hidden_smaller", 
            "model_dim": 8192,
            "hidden_dim": 2048
        }
    ]
    
    # JIT options - default is JIT only, add no-jit if requested
    jit_options = [{"name": "with_jit", "flag": []}]
    if args.include_no_jit:
        jit_options.append({"name": "no_jit", "flag": ["--no-jit"]})
    
    strategies = ["default", "megatron"]
    
    total_commands = len(configs) * len(jit_options) * len(strategies)
    command_count = 0
    
    for config in configs:
        print(f"Processing configuration: {config['name']} (model_dim={config['model_dim']}, hidden_dim={config['hidden_dim']})")
        
        for jit_option in jit_options:
            for strategy in strategies:
                command_count += 1
                trace_dir = f"tb_logs/{config['name']}/{jit_option['name']}/{strategy}"
                
                # Build the full command
                cmd_parts = [
                    f"rm -rf {trace_dir}",
                    "&&"
                ] + base_cmd + [
                    "--trace-dir", trace_dir,
                    "--model-dim", str(config['model_dim']),
                    "--hidden-dim", str(config['hidden_dim']),
                    "--sharding-strategy", strategy
                ] + jit_option['flag']
                
                cmd_str = " ".join(cmd_parts)
                
                print(f"[{command_count}/{total_commands}] Command: {cmd_str}")
                
                if args.dryrun:
                    print(f"  [DRY RUN] Would execute: {cmd_str}")
                else:
                    print(f"  [EXECUTING] {cmd_str}")
                    try:
                        # Execute the command
                        result = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
                        
                        if result.returncode == 0:
                            print("  ✓ Success")
                        else:
                            print("  ✗ Failed")
                            print(f"  Error: {result.stderr}")
                    except Exception as e:
                        print(f"  ✗ Exception: {e}")
                
                print()
    
    print("=== Profiling Complete ===")
    print("View results with: tensorboard --logdir=tb_logs")


if __name__ == "__main__":
    main()
0