#!/bin/bash
# Compute normalization statistics for all SpaceCIL task datasets.
# Run this before training to ensure norm stats are available.
#
# Usage:
#   ./scripts/compute_all_norm_stats.sh
#   ./scripts/compute_all_norm_stats.sh --max-frames 1000  # fast estimation

set -euo pipefail

CONFIGS=(spacecil_rm75_payload spacecil_rm75_latch spacecil_rm75_clean spacecil_rm75_connector)
EXTRA_ARGS=("$@")  # Pass through any additional arguments

for config in "${CONFIGS[@]}"; do
    echo "Computing norm stats for $config..."
    uv run scripts/compute_norm_stats.py --config-name "$config" "${EXTRA_ARGS[@]}"
done

echo "All norm stats computed successfully."
