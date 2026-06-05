# Isaac Sim 环境变量与启动别名
# 用法: source scripts/isaac_env.sh
#
# 重要: __GL_SHADER_DISK_CACHE=0 是当前驱动(580.65.06) +
#       Blackwell GPU(RTX 5060 Ti) 的必要缓解措施。
#       禁用 GL 着色器磁盘缓存可避免 libnvidia-gpucomp.so
#       在 GPU 着色器编译期间崩溃。

# 激活 openpi 环境（策略服务器）
alias openpi-env='source /home/srh/VLA/openpi/.venv/bin/activate'

# 激活 Isaac Sim 环境（含 Blackwell GPU 修复）
alias isaac-env='export __GL_SHADER_DISK_CACHE=0 && source /home/srh/VLA/openpi/.venv-isaac/bin/activate'