#!/usr/bin/env uv run python
"""
Timing sweep script for FeedForward inference across different sequence lengths.
Compares default (FSDP) vs megatron (tensor parallel) sharding strategies.

Usage: python timing_sweep.py --output-dir /path/to/results
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import numpy as np


def parse_timing_output(stdout_text):
    """Parse timing output to extract raw times and calculate statistics."""
    # Look for the pattern: Times (raw ms): [list of times]
    pattern = r"Times \(raw ms\): \[([\d\., ]+)\]"
    match = re.search(pattern, stdout_text)
    
    if not match:
        return None, None, None
    
    # Extract the times string and convert to list of floats
    times_str = match.group(1)
    times = [float(x.strip()) for x in times_str.split(',')]
    
    # Calculate statistics
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    return times, mean_time, std_time


def run_timing_test(seq_len, sharding_strategy, batch_size=2, num_shards=2, model_dim=2048, hidden_dim=8192, dryrun=False):
    """Run a single timing test and return results."""
    print(f"Running: seq_len={seq_len}, strategy={sharding_strategy}")
    
    # Build command
    cmd = [
        "uv", "run", "scripts/kmalik2/test_feedforward_inference.py",
        "--mode", "timing",
        "--seq-len", str(seq_len),
        "--sharding-strategy", sharding_strategy,
        "--batch-size", str(batch_size),
        "--num-shards", str(num_shards),
        "--model-dim", str(model_dim),
        "--hidden-dim", str(hidden_dim),
        "--timing-runs", "10"
    ]
    
    # Print command for dry run or actual execution
    cmd_str = " ".join(cmd)
    print(f"  Command: {cmd_str}")
    
    if dryrun:
        print(f"  [DRY RUN] Would execute command above")
        return {
            'seq_len': seq_len,
            'sharding_strategy': sharding_strategy,
            'batch_size': batch_size,
            'num_shards': num_shards,
            'model_dim': model_dim,
            'hidden_dim': hidden_dim,
            'raw_times': [1.0] * 10,  # Dummy data
            'mean_time_ms': 1.0,
            'std_time_ms': 0.1,
            'wall_time_sec': 10.0,
            'stdout': "DRY RUN - No actual output",
            'stderr': ""
        }
    
    try:
        # Run the command and capture output
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        end_time = time.time()
        
        if result.returncode != 0:
            print(f"  ✗ Command failed with return code {result.returncode}")
            print(f"  Error: {result.stderr}")
            return None
        
        # Parse timing results
        raw_times, mean_time, std_time = parse_timing_output(result.stdout)
        
        if raw_times is None:
            print(f"  ✗ Could not parse timing results")
            return None
        
        print(f"  ✓ Success: {mean_time:.3f}ms ± {std_time:.3f}ms")
        
        return {
            'seq_len': seq_len,
            'sharding_strategy': sharding_strategy,
            'batch_size': batch_size,
            'num_shards': num_shards,
            'model_dim': model_dim,
            'hidden_dim': hidden_dim,
            'raw_times': raw_times,
            'mean_time_ms': mean_time,
            'std_time_ms': std_time,
            'wall_time_sec': end_time - start_time,
            'stdout': result.stdout,
            'stderr': result.stderr
        }
        
    except subprocess.TimeoutExpired:
        print(f"  ✗ Command timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"  ✗ Command failed with exception: {e}")
        return None


def save_detailed_output(result, output_dir):
    """Save detailed stdout/stderr to individual files."""
    seq_len = result['seq_len']
    strategy = result['sharding_strategy']
    
    # Create subdirectory for this configuration
    config_dir = Path(output_dir) / f"seq{seq_len}_{strategy}"
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Save stdout
    stdout_file = config_dir / "stdout.txt"
    with open(stdout_file, 'w') as f:
        f.write(result['stdout'])
    
    # Save stderr
    stderr_file = config_dir / "stderr.txt"
    with open(stderr_file, 'w') as f:
        f.write(result['stderr'])
    
    # Save summary
    summary_file = config_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Configuration:\n")
        f.write(f"  Sequence Length: {result['seq_len']}\n")
        f.write(f"  Sharding Strategy: {result['sharding_strategy']}\n")
        f.write(f"  Batch Size: {result['batch_size']}\n")
        f.write(f"  Num Shards: {result['num_shards']}\n")
        f.write(f"  Model Dim: {result['model_dim']}\n")
        f.write(f"  Hidden Dim: {result['hidden_dim']}\n")
        f.write(f"\nTiming Results:\n")
        f.write(f"  Mean Time: {result['mean_time_ms']:.4f}ms\n")
        f.write(f"  Std Dev: {result['std_time_ms']:.4f}ms\n")
        f.write(f"  Wall Time: {result['wall_time_sec']:.2f}s\n")
        f.write(f"  Raw Times: {result['raw_times']}\n")


def append_to_csv(result, csv_file, write_header=False):
    """Append result to CSV file."""
    fieldnames = [
        'seq_len', 'sharding_strategy', 'batch_size', 'num_shards', 
        'model_dim', 'hidden_dim', 'mean_time_ms', 'std_time_ms', 
        'wall_time_sec', 'raw_times'
    ]
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        # Prepare row data
        row_data = {
            'seq_len': result['seq_len'],
            'sharding_strategy': result['sharding_strategy'],
            'batch_size': result['batch_size'],
            'num_shards': result['num_shards'],
            'model_dim': result['model_dim'],
            'hidden_dim': result['hidden_dim'],
            'mean_time_ms': result['mean_time_ms'],
            'std_time_ms': result['std_time_ms'],
            'wall_time_sec': result['wall_time_sec'],
            'raw_times': str(result['raw_times'])  # Convert list to string
        }
        
        writer.writerow(row_data)


def main():
    parser = argparse.ArgumentParser(description="Timing sweep for FeedForward sharding strategies")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save results")
    parser.add_argument("--dryrun", action="store_true",
                       help="Print commands without executing them")
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Batch size (default: 2)")
    parser.add_argument("--num-shards", type=int, default=2,
                       help="Number of shards (default: 2)")
    parser.add_argument("--model-dim", type=int, default=2048,
                       help="Model dimension (default: 2048)")
    parser.add_argument("--hidden-dim", type=int, default=8192,
                       help="Hidden dimension (default: 8192)")
    parser.add_argument("--min-seq-len", type=int, default=256,
                       help="Minimum sequence length (default: 256)")
    parser.add_argument("--max-seq-len", type=int, default=8192,
                       help="Maximum sequence length (default: 8192)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # CSV file for results - delete if it exists
    csv_file = output_dir / "timing_results.csv"
    if csv_file.exists():
        csv_file.unlink()
        print(f"Deleted existing CSV file: {csv_file}")
        print()
    
    # Generate sequence lengths (logarithmic)
    seq_lengths = []
    seq_len = args.min_seq_len
    while seq_len <= args.max_seq_len:
        seq_lengths.append(seq_len)
        seq_len *= 2
    
    # Sharding strategies to test
    strategies = ["default", "megatron"]
    
    print("=" * 60)
    print("FEEDFORWARD TIMING SWEEP")
    print("=" * 60)
    print(f"Mode: {'DRY RUN' if args.dryrun else 'EXECUTION'}")
    print(f"Output directory: {output_dir}")
    print(f"Sequence lengths: {seq_lengths}")
    print(f"Sharding strategies: {strategies}")
    print(f"Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num shards: {args.num_shards}")
    print(f"  Model dim: {args.model_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print()
    
    # Run all combinations
    total_runs = len(seq_lengths) * len(strategies)
    current_run = 0
    successful_runs = 0
    failed_runs = 0
    first_result = True  # Track if we need to write CSV header
    
    for seq_len in seq_lengths:
        for strategy in strategies:
            current_run += 1
            print(f"[{current_run}/{total_runs}] Testing seq_len={seq_len}, strategy={strategy}")
            
            result = run_timing_test(
                seq_len=seq_len,
                sharding_strategy=strategy,
                batch_size=args.batch_size,
                num_shards=args.num_shards,
                model_dim=args.model_dim,
                hidden_dim=args.hidden_dim,
                dryrun=args.dryrun
            )
            
            if result is not None:
                # Save detailed output
                save_detailed_output(result, output_dir)
                
                # Append to CSV (write header for first result)
                append_to_csv(result, csv_file, write_header=first_result)
                if first_result:
                    first_result = False
                
                successful_runs += 1
                print(f"  Results saved to {output_dir}")
            else:
                failed_runs += 1
            
            print()
    
    # Final summary
    print("=" * 60)
    print(f"TIMING SWEEP {'DRY RUN ' if args.dryrun else ''}COMPLETED")
    print("=" * 60)
    print(f"Total runs: {total_runs}")
    if args.dryrun:
        print(f"Commands generated: {successful_runs}")
        print(f"Would save results to: {output_dir}")
        print(f"Would create CSV file: {csv_file}")
        print(f"\nTo execute for real, remove --dryrun flag")
    else:
        print(f"Successful: {successful_runs}")
        print(f"Failed: {failed_runs}")
        print(f"Results saved to: {output_dir}")
        print(f"CSV file: {csv_file}")
        
        if successful_runs > 0:
            print(f"\nTo view results:")
            print(f"  cat {csv_file}")
            print(f"  ls {output_dir}/*/")


if __name__ == "__main__":
    main()
