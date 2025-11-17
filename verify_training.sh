#!/bin/bash
# Quick verification script for 2x RTX 3090 multi-GPU training

# Set environment variables
export HF_DATASETS_CACHE=/data/jerry/datasets/openx/cache/datasets
# export HF_HOME=/data/jerry/datasets/openx/cache
export HF_LEROBOT_HOME=/data/jerry/datasets/openx

echo "=== Step 1: Verifying normalization stats exist ==="
if [ -f "assets/pi0_bridge_finetune/bridge_lerobot_224/norm_stats.json" ]; then
    echo "✓ Normalization stats found"
else
    echo "✗ Normalization stats NOT found. Run compute_norm_stats.py first!"
    exit 1
fi

echo ""
echo "=== Step 2: Testing 2-GPU training (10 steps) ==="
# Use batch-size 2 (1 per GPU) and num-workers 0 for minimal memory usage
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name test_2gpu_verification \
    --num-train-steps 10 \
    --batch-size 2 \
    --num-workers 0 \
    --save-interval 5

echo ""
echo "=== Step 3: Verifying checkpoint was created ==="
if [ -d "checkpoints/pi0_bridge_finetune/test_2gpu_verification/params/checkpoint_5" ]; then
    echo "✓ Checkpoint created successfully"
else
    echo "✗ Checkpoint NOT found"
    exit 1
fi

echo ""
echo "=== Step 4: Testing resume from checkpoint (5 more steps) ==="
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train_pytorch.py pi0_bridge_finetune \
    --exp-name test_2gpu_verification \
    --resume \
    --num-train-steps 15 \
    --batch-size 2 \
    --num-workers 0

echo ""
echo "=== Step 5: Verifying final checkpoint ==="
if [ -d "checkpoints/pi0_bridge_finetune/test_2gpu_verification/params/checkpoint_10" ]; then
    echo "✓ Resume and checkpoint creation successful"
else
    echo "✗ Final checkpoint NOT found"
    exit 1
fi

echo ""
echo "======================================"
echo "✓ ALL VERIFICATION TESTS PASSED!"
echo "======================================"
echo ""
echo "Your setup is ready for full training on the cluster."
echo ""
echo "To clean up test files:"
echo "  rm -rf checkpoints/pi0_bridge_finetune/test_2gpu_verification"
