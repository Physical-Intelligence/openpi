export WANDB_MODE=offline
export HIP_VISIBLE_DEVICES=6
export HF_LEROBOT_HOME=/embodied/datasets/huggingface_vla/libero
export OPENPI_DATA_HOME=/embodied/models/Spatialtemporal-AI/openpi-assets
export OPENPI_PALIGEMMA_TOKENIZER_PATH=/embodied/models/Spatialtemporal-AI/openpi-assets/tokenizer/paligemma_tokenizer.model

# 单卡训练 + 性能分析（预热 5 step → 采集 5 step → 继续训练）
python scripts/train_pytorch_perf.py pi0_libero \
  --exp_name=my_libero_finetune \
  --save_interval=200 \
  --batch-size 104 \
  --overwrite \
  --perf
# Optional: fused joint attention (reduces aten::mm / aten::matmul)
#   --pytorch-use-joint-sdpa