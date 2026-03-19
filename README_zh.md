# openpi

openpi 是由 [Physical Intelligence 团队](https://www.physicalintelligence.company/) 发布的机器人开源模型与代码库。

目前，本仓库包含三种类型的模型：
- [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)：一种基于流匹配的视觉-语言-动作模型（VLA）。
- [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast)：一种基于 FAST 动作分词器的自回归 VLA 模型。
- [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)：π₀ 的升级版本，采用[知识隔离技术](https://www.physicalintelligence.company/research/knowledge_insulation)训练，具备更出色的开放世界泛化能力。请注意，在本仓库中，π₀.₅ 的训练和推理目前仅支持流匹配头部（flow matching head）。

对于所有模型，我们均提供在超过 1 万小时的机器人数据上预训练的**基础模型检查点**，以及开箱即用的示例和自定义数据微调指南。

这是一项实验性工作：π₀ 是为我们自己的机器人平台开发的，这些平台与广泛使用的 [ALOHA](https://tonyzhaozh.github.io/aloha/) 和 [DROID](https://droid-dataset.github.io/) 等主流平台存在差异。尽管我们乐观地认为，研究人员和从业者将能够开展创造性的新实验，将 π₀ 适配到各自的机器人平台，但我们并不期望每一次尝试都能取得成功。总而言之：π₀ 对您可能有效，也可能无效，但欢迎您尝试并亲自验证！

## 更新动态

- [2025年9月] 我们在 openpi 中发布了 PyTorch 支持。
- [2025年9月] 我们发布了 π₀.₅，这是 π₀ 的升级版本，具备更出色的开放世界泛化能力。
- [2025年9月] 我们为 DROID 训练添加了[改进的空闲过滤器](examples/droid/README_train.md#data-filtering)。
- [2025年6月] 我们添加了使用 `openpi` 在完整 [DROID 数据集](https://droid-dataset.github.io/)上训练 VLA 的[说明文档](examples/droid/README_train.md)。这是 pi0-FAST-DROID 训练流程的开源近似实现。


## 系统要求

运行本仓库中的模型需要具备以下最低规格的 NVIDIA GPU。以下估算基于单 GPU 场景，但您也可以通过在训练配置中设置 `fsdp_devices` 参数来使用模型并行技术，在多 GPU 环境下运行，从而降低单卡显存需求。另请注意，当前的训练脚本暂不支持多节点训练。

| 运行模式             | 显存需求    | 示例 GPU           |
| -------------------- | ----------- | ------------------ |
| 推理                 | > 8 GB      | RTX 4090           |
| 微调 (LoRA)          | > 22.5 GB   | RTX 4090           |
| 微调 (全参数)        | > 70 GB     | A100 (80GB) / H100 |

本仓库已在 Ubuntu 22.04 系统上完成测试，暂不支持其他操作系统。

## 安装指南

克隆本仓库时，请确保更新子模块：

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# 如果已经克隆了仓库，请执行：
git submodule update --init --recursive
```

我们使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖。请参阅 [uv 安装指南](https://docs.astral.sh/uv/getting-started/installation/)进行安装。安装 uv 后，执行以下命令配置环境：

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**注意**：需要设置 `GIT_LFS_SKIP_SMUDGE=1` 环境变量以正确拉取 LeRobot 依赖项。

**Docker 安装**：作为 uv 安装的替代方案，我们提供了使用 Docker 安装 openpi 的详细说明。如果您在系统配置过程中遇到问题，建议使用 Docker 简化安装流程。详情请参阅 [Docker 配置指南](docs/docker.md)。




## 模型检查点

### 基础模型
我们提供多个基础 VLA 模型检查点。这些检查点已在超过 1 万小时的机器人数据上完成预训练，可用于微调。

| 模型         | 用途       | 描述                                                                                                       | 检查点路径                                    |
| ------------ | ---------- | ---------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| $\pi_0$      | 微调       | 用于微调的基础 [π₀ 模型](https://www.physicalintelligence.company/blog/pi0)                               | `gs://openpi-assets/checkpoints/pi0_base`     |
| $\pi_0$-FAST | 微调       | 用于微调的基础自回归 [π₀-FAST 模型](https://www.physicalintelligence.company/research/fast)               | `gs://openpi-assets/checkpoints/pi0_fast_base`|
| $\pi_{0.5}$  | 微调       | 用于微调的基础 [π₀.₅ 模型](https://www.physicalintelligence.company/blog/pi05)                            | `gs://openpi-assets/checkpoints/pi05_base`    |

### 微调模型
我们还为各种机器人平台和任务提供"专家"检查点。这些模型是在上述基础模型基础上微调而成，可直接在目标机器人上运行。这些模型在您的特定机器人上可能有效，也可能无效。由于这些检查点是在相对较小的数据集上微调的，这些数据集使用的是更为广泛可用的机器人（如 ALOHA 和 DROID Franka 配置）收集，因此它们可能无法泛化到您的特定配置。不过，我们在实践中发现，其中部分模型（尤其是 DROID 检查点）具有相当广泛的泛化能力。

| 模型                    | 用途           | 描述                                                                                                                                                                                              | 检查点路径                                           |
| ----------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| $\pi_0$-FAST-DROID      | 推理           | 在 [DROID 数据集](https://droid-dataset.github.io/)上微调的 $\pi_0$-FAST 模型：可在 DROID 机器人平台上对新场景中的各类简单桌面操作任务进行零样本执行                                                                 | `gs://openpi-assets/checkpoints/pi0_fast_droid`      |
| $\pi_0$-DROID           | 微调           | 在 [DROID 数据集](https://droid-dataset.github.io/)上微调的 $\pi_0$ 模型：推理速度比 $\pi_0$-FAST-DROID 更快，但对语言指令的跟随能力可能稍弱                                                        | `gs://openpi-assets/checkpoints/pi0_droid`           |
| $\pi_0$-ALOHA-towel     | 推理           | 在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调的 $\pi_0$ 模型：可在 ALOHA 机器人平台上零样本折叠各种毛巾                                                                           | `gs://openpi-assets/checkpoints/pi0_aloha_towel`     |
| $\pi_0$-ALOHA-tupperware| 推理           | 在内部 [ALOHA](https://tonyzhaozh.github.io/aloha/) 数据上微调的 $\pi_0$ 模型：可从保鲜盒容器中取出食物                                                                                             | `gs://openpi-assets/checkpoints/pi0_aloha_tupperware`|
| $\pi_0$-ALOHA-pen-uncap | 推理           | 在公开 [ALOHA](https://dit-policy.github.io/) 数据上微调的 $\pi_0$ 模型：可拔掉笔帽                                                                                                               | `gs://openpi-assets/checkpoints/pi0_aloha_pen_uncap` |
| $\pi_{0.5}$-LIBERO      | 推理           | 为 [LIBERO](https://libero-project.github.io/datasets) 基准测试微调的 $\pi_{0.5}$ 模型：达到最先进性能（参见 [LIBERO README](examples/libero/README.md)）                                          | `gs://openpi-assets/checkpoints/pi05_libero`         |
| $\pi_{0.5}$-DROID       | 推理 / 微调    | 在 [DROID 数据集](https://droid-dataset.github.io/)上采用[知识隔离技术](https://www.physicalintelligence.company/research/knowledge_insulation)微调的 $\pi_{0.5}$ 模型：推理速度快且语言跟随能力强 | `gs://openpi-assets/checkpoints/pi05_droid`         |


默认情况下，检查点会自动从 `gs://openpi-assets` 下载，并缓存至 `~/.cache/openpi` 目录。您可以通过设置 `OPENPI_DATA_HOME` 环境变量来覆盖下载路径。





## 运行预训练模型推理

我们的预训练模型检查点只需几行代码即可运行（以下展示 $\pi_0$-FAST-DROID 模型示例）：
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

# 创建训练好的策略
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 在示例数据上运行推理
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    ...
    "prompt": "拿起叉子"
}
action_chunk = policy.infer(example)["actions"]
```
您也可以在[示例笔记本](examples/inference.ipynb)中进行测试。

我们为在 [DROID](examples/droid/README.md) 和 [ALOHA](examples/aloha_real/README.md) 机器人上运行预训练检查点推理提供了详细的分步示例。

**远程推理**：我们提供了用于**远程**运行模型推理的[示例和代码](docs/remote_inference.md)：模型可在不同的服务器上运行，并通过 WebSocket 连接向机器人流式传输动作。这便于在机器人外使用更强大的 GPU，并将机器人环境与策略环境分离。

**无机器人测试推理**：我们提供了一个用于在没有机器人的情况下测试推理的[脚本](examples/simple_client/README.md)。该脚本将生成随机观测数据并使用模型运行推理。详情请参阅[此处](examples/simple_client/README.md)。





## 在自定义数据上微调基础模型

我们将以在 [LIBERO 数据集](https://libero-project.github.io/datasets)上微调 $\pi_{0.5}$ 模型为例，演示如何在自定义数据上微调基础模型。我们将介绍三个步骤：
1. 将数据转换为 LeRobot 数据集格式（我们用于训练的格式）
2. 定义训练配置并运行训练
3. 启动策略服务器并运行推理

### 1. 将数据转换为 LeRobot 数据集

我们在 [`examples/libero/convert_libero_data_to_lerobot.py`](examples/libero/convert_libero_data_to_lerobot.py) 中提供了将 LIBERO 数据转换为 LeRobot 数据集格式的最小示例脚本。您可以轻松修改此脚本以转换自己的数据！您可以从[此处](https://huggingface.co/datasets/openvla/modified_libero_rlds)下载原始 LIBERO 数据集，然后运行以下脚本：

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/libero/data
```

**注意**：如果您只想在 LIBERO 上进行微调，可以跳过此步骤，因为我们的 LIBERO 微调配置已指向预转换的 LIBERO 数据集。此步骤仅作为您适配自己数据的示例。

### 2. 定义训练配置并运行训练

要在自定义数据上微调基础模型，您需要定义数据处理和训练的配置。我们为 LIBERO 提供了带有详细注释的示例配置，您可以针对自己的数据集进行修改：

- [`LiberoInputs` 和 `LiberoOutputs`](src/openpi/policies/libero_policy.py)：定义 LIBERO 环境与模型之间的数据映射关系，用于训练和推理。
- [`LeRobotLiberoDataConfig`](src/openpi/training/config.py)：定义如何从 LeRobot 数据集处理 LIBERO 原始数据用于训练。
- [`TrainConfig`](src/openpi/training/config.py)：定义微调超参数、数据配置和权重加载器。

我们为 LIBERO 数据上的 [π₀](src/openpi/training/config.py)、[π₀-FAST](src/openpi/training/config.py) 和 [π₀.₅](src/openpi/training/config.py) 提供了示例微调配置。

在运行训练之前，我们需要计算训练数据的归一化统计量。使用您的训练配置名称运行以下脚本：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

现在我们可以启动训练，命令如下（`--overwrite` 标志用于在使用相同配置重新运行微调时覆盖现有检查点）：

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite
```

该命令会将训练进度输出到控制台，并将检查点保存到 `checkpoints` 目录。您还可以在 Weights & Biases 仪表板上监控训练进度。为了最大化利用 GPU 显存，请在运行训练前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` —— 这使 JAX 能够使用高达 90% 的 GPU 显存（默认值为 75%）。

**注意**：我们提供了从预训练中*重载*状态/动作归一化统计量的功能。如果您正在微调的机器人是我们预训练混合数据中的一部分，这可能会有所帮助。有关如何重载归一化统计量的更多详情，请参阅 [norm_stats.md](docs/norm_stats.md) 文件。

### 3. 启动策略服务器并运行推理

训练完成后，我们可以通过启动策略服务器，然后从 LIBERO 评估脚本查询它来运行推理。启动模型服务器非常简单（本示例使用第 20,000 次迭代的检查点，请根据需要修改）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

这将启动一个监听 8000 端口的服务器，等待接收观测数据。然后我们可以运行评估脚本（或机器人运行时）来查询服务器。

对于 LIBERO 评估的运行，我们提供（并推荐使用）Docker 化工作流程，可同时处理策略服务器和评估脚本。详情请参阅 [LIBERO README](examples/libero/README.md)。

如果您想在自己的机器人运行时中嵌入策略服务器调用，我们在[远程推理文档](docs/remote_inference.md)中提供了一个最小示例。



### 更多示例

我们在以下 README 中提供了更多在 ALOHA 平台上微调和运行模型推理的示例：
- [ALOHA 仿真环境](examples/aloha_sim)
- [ALOHA 真实机器人](examples/aloha_real)
- [UR5](examples/ur5)

## PyTorch 支持

openpi 现已提供 π₀ 和 π₀.₅ 模型的 PyTorch 实现，与原始 JAX 版本并存！PyTorch 实现已在 LIBERO 基准测试上完成了验证（包括推理和微调）。以下功能目前暂不支持（未来可能会有所变化）：

- π₀-FAST 模型
- 混合精度训练
- FSDP（完全分片数据并行）训练
- LoRA（低秩自适应）训练
- 训练过程中的 EMA（指数移动平均）权重

### 环境配置
1. 确保已安装所有依赖项的最新版本：`uv sync`

2. 确认已安装 transformers 4.53.2 版本：`uv pip show transformers`

3. 应用 transformers 库补丁：
   ```bash
   cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/
   ```

这将使用必要的模型更改覆盖 transformers 库中的多个文件：1) 支持 AdaRMS，2) 正确控制激活精度，以及 3) 允许在不更新的情况下使用 KV 缓存。

**警告**：使用默认的 uv 链接模式（硬链接）时，这将永久影响 uv 缓存中的 transformers 库，这意味着这些更改将在 transformers 重新安装后仍然存在，甚至可能传播到其他使用 transformers 的项目。要完全撤销此操作，您必须运行 `uv cache clean transformers`。

### 将 JAX 模型转换为 PyTorch

要将 JAX 模型检查点转换为 PyTorch 格式：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /path/to/jax/checkpoint \
    --config_name <config name> \
    --output_path /path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 运行推理

PyTorch 实现使用与 JAX 版本相同的 API —— 您只需将检查点路径更改为转换后的 PyTorch 模型：

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = "/path/to/converted/pytorch/checkpoint"

# 创建训练好的策略（自动检测 PyTorch 格式）
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 运行推理（API 与 JAX 相同）
action_chunk = policy.infer(example)["actions"]
```

### 使用 PyTorch 运行策略服务器

策略服务器与 PyTorch 模型配合使用方式相同 —— 只需指向转换后的检查点目录：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_droid \
    --policy.dir=/path/to/converted/pytorch/checkpoint
```

### 使用 PyTorch 进行微调

要在 PyTorch 中微调模型：

1. 将 JAX 基础模型转换为 PyTorch 格式：
   ```bash
   uv run examples/convert_jax_model_to_pytorch.py \
       --config_name <config name> \
       --checkpoint_dir /path/to/jax/base/model \
       --output_path /path/to/pytorch/base/model
   ```

2. 在配置中使用 `pytorch_weight_path` 指定转换后的 PyTorch 模型路径

3. 使用以下模式之一启动训练：

```bash
# 单 GPU 训练：
uv run scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>

# 示例：
uv run scripts/train_pytorch.py debug --exp_name pytorch_test
uv run scripts/train_pytorch.py debug --exp_name pytorch_test --resume  # 从最新检查点恢复

# 多 GPU 训练（单节点）：
uv run torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>

# 示例：
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume

# 多节点训练：
uv run torchrun \
    --nnodes=<num_nodes> \
    --nproc_per_node=<gpus_per_node> \
    --node_rank=<rank_of_node> \
    --master_addr=<master_ip> \
    --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>
```

### 精度设置

JAX 和 PyTorch 实现的精度处理方式如下：

**JAX：**
1. 推理：大部分权重和计算使用 bfloat16，少数计算使用 float32 以保证稳定性
2. 训练：默认使用混合精度：权重和梯度为 float32，（大部分）激活和计算为 bfloat16。您可以在配置中将 `dtype` 设置为 float32 来切换到全 float32 训练。

**PyTorch：**
1. 推理：与 JAX 一致 —— 大部分权重和计算使用 bfloat16，少数权重转换为 float32 以保证稳定性
2. 训练：支持全 bfloat16（默认）或全 float32。您可以在配置中设置 `pytorch_training_precision` 来更改。bfloat16 占用更少内存，但相比 float32 会有更高的损失值。混合精度暂不支持。

借助 torch.compile，PyTorch 的推理速度与 JAX 相当。

## 故障排除

我们在此收集常见问题及其解决方案。如果遇到问题，请先查阅此处。如果找不到解决方案，请在仓库提交 issue（参见[此处](CONTRIBUTING.md)获取指南）。

| 问题                                     | 解决方案                                                                                                                                                                                   |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `uv sync` 因依赖冲突失败                 | 尝试删除虚拟环境目录（`rm -rf .venv`）后重新运行 `uv sync`。如果问题持续，请检查是否安装了最新版本的 `uv`（`uv self update`）。                                                              |
| 训练时 GPU 显存不足                      | 确保在运行训练前设置 `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`（或更高），以允许 JAX 使用更多 GPU 显存。您还可以使用 `--fsdp-devices <n>`（其中 `<n>` 为 GPU 数量）启用[完全分片数据并行](https://engineering.fb.com/2021/07/15/open-source/fsdp/)，这会以降低训练速度为代价减少显存占用（具体降低程度取决于您的配置）。如果仍然显存不足，您可能需要考虑禁用 EMA。        |
| 策略服务器连接错误                       | 检查服务器是否正在运行并监听预期端口。验证客户端与服务器之间的网络连接和防火墙设置。                                                                                                         |
| 训练时缺少 norm stats 错误               | 在开始训练前，使用您的配置名称运行 `scripts/compute_norm_stats.py`。                                                                                                                        |
| 数据集下载失败                           | 检查您的网络连接。对于 HuggingFace 数据集，请确保已登录（`huggingface-cli login`）。                                                                                                        |
| CUDA/GPU 错误                            | 验证 NVIDIA 驱动程序已正确安装。对于 Docker，请确保已安装 nvidia-container-toolkit。检查 GPU 兼容性。您无需在系统级别安装 CUDA 库 —— 它们将通过 uv 安装。如果遇到 CUDA 问题，您甚至可以尝试*卸载*系统 CUDA 库，因为系统库有时会导致冲突。 |
| 运行示例时出现导入错误                   | 确保已使用 `uv sync` 安装所有依赖项。某些示例可能在其 README 中列出了额外的要求。                                                                                                          |
| 动作维度不匹配                           | 验证您的数据处理转换是否与机器人的预期输入/输出维度匹配。检查策略类中的动作空间定义。                                                                                                       |
| 训练损失发散                             | 检查数据集 `norm_stats.json` 中的 `q01`、`q99` 和 `std` 值。某些很少使用的维度最终可能具有非常小的 `q01`、`q99` 或 `std` 值，导致归一化后的状态和动作变得非常大。您可以手动调整 norm stats 作为临时解决方案。 |