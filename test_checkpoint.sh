#!/bin/bash

# 设置检查点路径
CHECKPOINT_DIR="checkpoints/pi05_aloha_sim_insertion_human/epri_aloha_sim_insertion_pi0520251208151450/1000"

# 正确设置TASK变量（注意等号两边不能有空格）
TASK="gym_aloha/AlohaInsertion-v0"
# TASK="gym_aloha/AlohaTransferCube-v0"

# 设置绝对输出目录路径
OUT_DIR="$(pwd)/data/aloha_sim/videos"

# 根据TASK自动生成对应的policy.config
case "$TASK" in
  "gym_aloha/AlohaInsertion-v0")
    POLICY_CONFIG="pi05_aloha_sim_insertion_human"
    ;;
  "gym_aloha/AlohaTransferCube-v0")
    POLICY_CONFIG="pi05_aloha_sim_transfer_cube_human"
    ;;
  *)
    echo "Unknown task: $TASK"
    exit 1
    ;;
esac

echo "Testing checkpoint from: ${CHECKPOINT_DIR}"
echo "Using task: ${TASK}"
echo "Using policy config: ${POLICY_CONFIG}"
echo "Output directory: ${OUT_DIR}"

# 创建输出目录（如果不存在）
mkdir -p "${OUT_DIR}"

# 启动策略服务器作为后台进程
echo "Starting policy server..."
python scripts/serve_policy.py policy:checkpoint \
  --policy.config="${POLICY_CONFIG}" \
  --policy.dir="${CHECKPOINT_DIR}" &

# 保存策略服务器的进程ID
POLICY_PID=$!

# 等待几秒钟让服务器启动
sleep 30

# 启动模拟环境
echo "Starting simulation environment..."
# 使用tyro的正确参数格式
python examples/aloha_sim/main.py --args.out-dir="${OUT_DIR}" --args.task="${TASK}"

# 当模拟环境结束后，终止策略服务器进程
echo "Terminating policy server..."
kill $POLICY_PID 2>/dev/null

echo "Test completed."