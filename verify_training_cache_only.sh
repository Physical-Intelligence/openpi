#!/bin/bash
# Verification script to test training with ONLY the HuggingFace datasets cache
# This verifies that the original LeRobot Parquet dataset is NOT needed

set -e  # Exit on any error

echo "======================================"
echo "Cache-Only Training Verification"
echo "======================================"
echo ""

# Set environment variables to use ONLY the cache
export HF_DATASETS_CACHE=/data/jerry/datasets/openx/cache/datasets
# export HF_HOME=/data/jerry/datasets/openx/cache
# Point to a NON-EXISTENT location to ensure we're not using the original dataset
export HF_LEROBOT_HOME=/tmp/nonexistent_lerobot_$(date +%s)

echo "=== Step 0: Create minimal dataset structure with metadata only ==="
# Create a temporary directory with ONLY metadata (no data/ directory)
TEMP_LEROBOT_HOME="/tmp/lerobot_meta_only_$(date +%s)"
mkdir -p "$TEMP_LEROBOT_HOME/bridge_lerobot_224"

# Copy only the metadata directory (92MB), not the data directory (152GB)
if [ -d "/data/jerry/datasets/openx/bridge_lerobot_224/meta" ]; then
    cp -r /data/jerry/datasets/openx/bridge_lerobot_224/meta "$TEMP_LEROBOT_HOME/bridge_lerobot_224/"
    META_SIZE=$(du -sh "$TEMP_LEROBOT_HOME/bridge_lerobot_224/meta" | cut -f1)
    echo "✓ Copied metadata to temp location: $TEMP_LEROBOT_HOME/bridge_lerobot_224/meta"
    echo "  Metadata size: $META_SIZE"
else
    echo "✗ Original metadata not found"
    exit 1
fi

# Explicitly verify data/ directory does NOT exist in temp location
if [ -d "$TEMP_LEROBOT_HOME/bridge_lerobot_224/data" ]; then
    echo "✗ ERROR: data/ directory should not exist"
    exit 1
else
    echo "✓ Confirmed: data/ directory does NOT exist (will use cache only)"
fi

export HF_LEROBOT_HOME="$TEMP_LEROBOT_HOME"
# Enable cache-only mode (our custom implementation)
export LEROBOT_USE_CACHE_ONLY=1

echo ""
echo "=== Configuration ==="
echo "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "HF_HOME: $HF_HOME"
echo "HF_LEROBOT_HOME: $HF_LEROBOT_HOME (metadata only, no data/)"
echo "LEROBOT_USE_CACHE_ONLY: $LEROBOT_USE_CACHE_ONLY"
echo ""

echo "=== Step 1: Verify cache exists ==="
if [ -d "$HF_DATASETS_CACHE" ]; then
    CACHE_SIZE=$(du -sh "$HF_DATASETS_CACHE" | cut -f1)
    ARROW_FILES=$(find "$HF_DATASETS_CACHE" -name "*.arrow" | wc -l)
    echo "✓ Cache directory found: $HF_DATASETS_CACHE"
    echo "  Size: $CACHE_SIZE"
    echo "  Arrow files: $ARROW_FILES"
else
    echo "✗ Cache directory NOT found at $HF_DATASETS_CACHE"
    echo "  Please run verify_training.sh first to create the cache"
    exit 1
fi

echo ""
echo "=== Step 2: Verify normalization stats exist ==="
if [ -f "assets/pi0_bridge_finetune/bridge_lerobot_224/norm_stats.json" ]; then
    echo "✓ Normalization stats found"
else
    echo "✗ Normalization stats NOT found"
    echo "  Run: uv run python scripts/compute_norm_stats.py --config-name=pi0_bridge_finetune"
    exit 1
fi

echo ""
echo "=== Step 3: Test training with cache only (10 steps) ==="
echo "This will FAIL if the original dataset is required..."
echo ""

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name test_cache_only_verification \
    --num-train-steps 10 \
    --batch-size 2 \
    --num-workers 0 \
    --save-interval 5

echo ""
echo "=== Step 4: Verify checkpoint was created ==="
if [ -d "checkpoints/pi0_bridge_finetune/test_cache_only_verification/params/checkpoint_5" ]; then
    echo "✓ Checkpoint created successfully using CACHE ONLY!"
else
    echo "✗ Checkpoint NOT found"
    exit 1
fi

echo ""
echo "=== Step 5: Test resume from checkpoint (5 more steps) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name test_cache_only_verification \
    --resume \
    --num-train-steps 15 \
    --batch-size 2 \
    --num-workers 0

echo ""
echo "=== Step 6: Verify final checkpoint ==="
if [ -d "checkpoints/pi0_bridge_finetune/test_cache_only_verification/params/checkpoint_10" ]; then
    echo "✓ Resume and checkpoint creation successful"
else
    echo "✗ Final checkpoint NOT found"
    exit 1
fi

echo ""
echo "=== Step 7: Cleanup temporary metadata ==="
rm -rf "$TEMP_LEROBOT_HOME"
echo "✓ Removed temporary directory: $TEMP_LEROBOT_HOME"

echo ""
echo "======================================"
echo "✓ SUCCESS: Training works with METADATA + CACHE!"
echo "======================================"
echo ""
echo "IMPORTANT FINDINGS:"
echo "  ✓ Metadata files are REQUIRED: meta/ directory (92MB)"
echo "  ✓ HuggingFace Arrow cache is REQUIRED: cache/ directory (152GB)"
echo "  ✓ Parquet data files are NOT needed: data/ directory (152GB) ← CAN DELETE"
echo ""
echo "Storage optimization:"
echo "  • Keep: /data/jerry/datasets/openx/bridge_lerobot_224/meta/ (92MB)"
echo "  • Keep: /data/jerry/datasets/openx/cache/ (152GB)"
echo "  • DELETE: /data/jerry/datasets/openx/bridge_lerobot_224/data/ (152GB) ← SAVES 152GB"
echo ""
echo "To delete the Parquet data directory:"
echo "  rm -rf /data/jerry/datasets/openx/bridge_lerobot_224/data/"
echo ""
echo "For cluster deployment, upload:"
echo "  1. The metadata: bridge_lerobot_224/meta/ (92MB)"
echo "  2. The HF cache: cache/datasets/ (152GB)"
echo "  Total upload: ~152GB instead of 304GB"
echo ""
echo "To clean up test files:"
echo "  rm -rf checkpoints/pi0_bridge_finetune/test_cache_only_verification"
echo ""
