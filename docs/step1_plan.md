# Step 1: 拆分推理管线 — 任务规划文档

> 状态：规划阶段
> 目标：为缓存系统的 CP1/CP2/CP3 拦截点提供类型化的公共接口，同时**零侵入**现有推理路径和计时系统

---

## 1. 现状分析（代码阅读结论）

### 1.1 已有的私有 Stage 拆分

`pi0_pytorch.py` 中已经存在三个私有方法：

```python
_stage1_token_prep(observation)
    → (state, prefix_embs, prefix_pad_masks, prefix_att_2d_masks_4d, prefix_position_ids)

_stage2_llm_backbone(prefix_embs, prefix_pad_masks, prefix_att_2d_masks_4d, prefix_position_ids)
    → past_key_values

_stage3_action_expert(state, prefix_pad_masks, past_key_values, noise, num_steps=10)
    → x_t  (action_chunk)
```

`sample_actions()` 已经通过这三个私有方法串联执行，不需要重写其逻辑。

`policy.py` 已通过 `hasattr(model, "_stage1_token_prep")` 检测并使用这三个私有方法做**按阶段计时**，将结果写入 `stage_timing` 字段。

### 1.2 现有计时系统

`policy.py` 中的现有计时系统：
- 使用 `time.monotonic()` 计时（CPU 时间）
- 在 CUDA 推理后调用 `torch.cuda.synchronize()` 对齐 GPU 操作
- 输出 `stage_timing = {token_prep_ms, llm_backbone_ms, action_expert_ms, total_ms}`

**Step 1 不得修改此系统**。缓存系统的精细计时将在 Step 2（`SystemTimer`）中实现。

### 1.3 缺失的内容（Step 1 需要补充）

| 缺失内容 | 用途 |
|---------|------|
| `Stage1Output` / `Stage2Output` / `Stage3Output` 数据类 | 阶段间传递数据，供 `InferenceInterceptor` 使用 |
| `run_stage1()` / `run_stage2()` / `run_stage3()` 公共方法 | 缓存系统的拦截点 |
| `run_stage3()` 的 `return_intermediates` 参数 | 保存流匹配中间状态 x_t，用于 Warm Start |
| `run_stage3_from(start_x, start_t)` | Warm Start 从中间状态继续去噪 |

---

## 2. Pi0 与 Pi0.5 的关键区别

在拆分时必须正确处理两种模型的差异：

### Stage 1（视觉编码 + Prefix 嵌入）

| 方面 | Pi0 (`pi05=False`) | Pi0.5 (`pi05=True`) |
|------|-------------------|---------------------|
| 状态输入 | `state` 是连续向量，在 Stage 3 的 Suffix 中处理 | `state` 已被变换管线离散化为文本 token，编码进 `lang_tokens` |
| Prefix 内容 | 图像 token + 语言 token | 图像 token + 语言 token（其中已含离散化状态） |
| `Stage1Output.state` | 原始连续向量，用于 Stage 3 的 `state_proj` | 原始连续向量，**不**进入模型，仅作为缓存键 |

> 关键：两种模型 `_stage1_token_prep` 逻辑相同，差异在下游（`embed_suffix` 分支）。

### Stage 2（LLM Backbone / KV Cache 填充）

- **两种模型逻辑相同**：均调用 PaliGemma 做前向传播并返回 `past_key_values`
- PyTorch 实现中 Pi0.5 **没有**自回归子任务生成（JAX 路径已禁用）
- `past_key_values` 是 HuggingFace `DynamicCache` 对象，**必须原样传递，不能 clone/reshape**

### Stage 3（Action Expert / 流匹配去噪）

| 方面 | Pi0 | Pi0.5 |
|------|-----|-------|
| Suffix token 数量 | 1（state）+ 50（action）= 51 | 50（action only） |
| 时间步注入 | 时间 embedding 与 action 拼接后经 MLP | 时间 embedding 经 `time_mlp` 后作为 adaRMSNorm 条件 |
| `adarms_cond` | `None` | 时间 MLP 输出的向量 |

> 去噪循环本身（Euler 步进）对两种模型相同，差异全部在 `denoise_step → embed_suffix` 内部，不影响 Stage 3 的接口设计。

---

## 3. 编码方案

### 3.1 文件改动范围

| 文件 | 改动内容 | 侵入程度 |
|------|---------|---------|
| `src/openpi/models_pytorch/pi0_pytorch.py` | 添加 3 个 dataclass + 4 个公共方法 | **仅添加**，不修改任何已有代码 |
| `src/openpi/policies/policy.py` | **不改动** | 零 |
| `src/openpi/models_pytorch/gemma_pytorch.py` | **不改动** | 零 |
| 其他所有文件 | **不改动** | 零 |

### 3.2 数据类接口设计

放置位置：`pi0_pytorch.py` 文件顶部（`import` 之后，`PI0Pytorch` 类之前）。

```python
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class Stage1Output:
    """Stage 1（视觉编码 + Prefix 准备）的输出。

    包含 Stage 2 所需的所有中间张量，以及缓存键构建所需的原始状态。
    张量均在 model 所在 device 上。
    """
    state: torch.Tensor
    # shape: [B, action_dim]
    # Pi0: 连续状态向量，将被 state_proj 投影到 suffix
    # Pi0.5: 连续状态向量，已在 prefix 中以 text token 形式存在；此处仅用于缓存键

    prefix_embs: torch.Tensor
    # shape: [B, prefix_len, emb_dim]
    # emb_dim = paligemma width (2048 for gemma_2b)

    prefix_pad_masks: torch.Tensor
    # shape: [B, prefix_len], dtype=torch.bool

    prefix_att_2d_masks_4d: torch.Tensor
    # shape: [B, 1, prefix_len, prefix_len], dtype=torch.float32
    # 已转换为 additive mask（0.0 / -inf）

    prefix_position_ids: torch.Tensor
    # shape: [B, prefix_len], dtype=torch.int64


@dataclass
class Stage2Output:
    """Stage 2（KV Cache 填充）的输出。

    包含 Stage 3 所需的所有内容，以及 Stage 1 的引用（方便 InferenceInterceptor 访问）。
    """
    stage1: Stage1Output
    past_key_values: Any
    # HuggingFace DynamicCache 对象（或 tuple of tuples）
    # 禁止 clone/deepcopy，必须原样传递给 denoise_step


@dataclass
class Stage3Output:
    """Stage 3（Action Expert 流匹配去噪）的输出。"""
    action_chunk: torch.Tensor
    # shape: [B, action_horizon, action_dim], dtype=torch.float32

    intermediates: Optional[dict[float, torch.Tensor]] = None
    # 仅当 return_intermediates=True 时填充
    # key = timestep 值（如 0.7, 0.5, 0.3），value = x_t at that timestep
    # shape of each value: [B, action_horizon, action_dim]
    # 表示"该时刻去噪步骤开始前"的 x_t，用于 Warm Start
```

### 3.3 公共方法接口设计

在 `PI0Pytorch` 类中添加以下方法（不带 `@torch.no_grad()` 装饰，由调用方控制上下文）：

#### `run_stage1`

```python
def run_stage1(self, observation) -> Stage1Output:
    """Stage 1：观测预处理 + SigLIP 视觉编码 + Prefix 嵌入。

    等价于现有 _stage1_token_prep()，但返回类型化的 Stage1Output。
    供缓存系统 InferenceInterceptor 使用，不影响 sample_actions() 路径。

    Args:
        observation: Observation 对象（与 sample_actions 接收的相同）

    Returns:
        Stage1Output，包含 prefix 相关张量和原始 state
    """
    state, prefix_embs, prefix_pad_masks, prefix_att_2d_masks_4d, prefix_position_ids = \
        self._stage1_token_prep(observation)
    return Stage1Output(
        state=state,
        prefix_embs=prefix_embs,
        prefix_pad_masks=prefix_pad_masks,
        prefix_att_2d_masks_4d=prefix_att_2d_masks_4d,
        prefix_position_ids=prefix_position_ids,
    )
```

#### `run_stage2`

```python
def run_stage2(self, stage1: Stage1Output) -> Stage2Output:
    """Stage 2：LLM Backbone 前向传播，填充 KV Cache。

    等价于现有 _stage2_llm_backbone()，但接收/返回类型化结构。
    Pi0 与 Pi0.5 逻辑相同。

    Args:
        stage1: run_stage1() 的输出

    Returns:
        Stage2Output，包含 past_key_values 和 stage1 引用
    """
    past_key_values = self._stage2_llm_backbone(
        stage1.prefix_embs,
        stage1.prefix_pad_masks,
        stage1.prefix_att_2d_masks_4d,
        stage1.prefix_position_ids,
    )
    return Stage2Output(stage1=stage1, past_key_values=past_key_values)
```

#### `run_stage3`

```python
def run_stage3(
    self,
    stage2: Stage2Output,
    *,
    noise: Optional[torch.Tensor] = None,
    num_steps: int = 10,
    return_intermediates: bool = False,
    save_timesteps: tuple[float, ...] = (0.7, 0.5, 0.3),
) -> Stage3Output:
    """Stage 3：Action Expert 全流程流匹配去噪（10 Euler 步）。

    当 return_intermediates=False 时（默认），直接委托给 _stage3_action_expert()，
    无任何额外开销，性能与原路径完全一致。

    当 return_intermediates=True 时，使用相同的 denoise_step() 实现独立循环，
    在选定时间步保存 x_t 副本，用于 Warm Start 缓存。

    Args:
        stage2: run_stage2() 的输出
        noise: 初始噪声，None 时自动采样
        num_steps: 去噪步数，默认 10
        return_intermediates: 是否保存中间状态
        save_timesteps: 要保存的时间步，仅在 return_intermediates=True 时生效

    Returns:
        Stage3Output，包含 action_chunk（和可选的 intermediates）
    """
    stage1 = stage2.stage1
    device = stage1.state.device
    bsize = stage1.state.shape[0]

    if noise is None:
        actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        noise = self.sample_noise(actions_shape, device)

    if not return_intermediates:
        # 零开销路径：直接调用已有私有方法
        action_chunk = self._stage3_action_expert(
            stage1.state,
            stage1.prefix_pad_masks,
            stage2.past_key_values,
            noise,
            num_steps,
        )
        return Stage3Output(action_chunk=action_chunk)
    else:
        action_chunk, intermediates = self._stage3_with_intermediates(
            stage1.state,
            stage1.prefix_pad_masks,
            stage2.past_key_values,
            noise,
            num_steps,
            save_timesteps,
        )
        return Stage3Output(action_chunk=action_chunk, intermediates=intermediates)
```

#### `run_stage3_from`（Warm Start）

```python
def run_stage3_from(
    self,
    stage2: Stage2Output,
    start_x: torch.Tensor,
    start_t: float,
    *,
    num_steps: int = 10,
) -> Stage3Output:
    """Warm Start：从缓存的中间状态 x_{start_t} 继续去噪。

    使用相同的 dt = -1/num_steps，从 start_t 步进到 0。
    步数 = round(start_t * num_steps)，例如：
      - start_t=0.3, num_steps=10 → 3 步（节省 70% Stage 3 计算）
      - start_t=0.5, num_steps=10 → 5 步（节省 50%）

    重要：start_x 必须是在与当前 observation 兼容的 stage2.past_key_values
    下生成的中间状态；使用"相似但不同"的 start_x 时存在精度损失，
    该损失的可接受范围将由 Step 7 实验确定。

    Args:
        stage2: run_stage2() 的输出（提供 state 和 past_key_values）
        start_x: 缓存的中间噪声张量，shape [B, action_horizon, action_dim]
        start_t: 起始时间步（如 0.3 表示从 t=0.3 开始）
        num_steps: 原始总步数（决定 dt）

    Returns:
        Stage3Output，包含最终 action_chunk
    """
    stage1 = stage2.stage1
    device = stage1.state.device
    bsize = stage1.state.shape[0]

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = start_x
    timestep = torch.tensor(start_t, dtype=torch.float32, device=device)

    while timestep >= -dt / 2:
        expanded_time = timestep.expand(bsize)
        v_t = self.denoise_step(
            stage1.state,
            stage1.prefix_pad_masks,
            stage2.past_key_values,
            x_t,
            expanded_time,
        )
        x_t = x_t + dt * v_t
        timestep += dt

    return Stage3Output(action_chunk=x_t)
```

#### `_stage3_with_intermediates`（私有辅助方法）

```python
def _stage3_with_intermediates(
    self,
    state: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    past_key_values,
    noise: torch.Tensor,
    num_steps: int,
    save_timesteps: tuple[float, ...],
) -> tuple[torch.Tensor, dict[float, torch.Tensor]]:
    """与 _stage3_action_expert 等价的去噪循环，额外在指定时间步保存 x_t。

    保存时机：在该时间步的 denoise_step() 调用**之前**保存，
    这样 intermediates[t] 对应"从 t 出发继续去噪"的起始点，
    与 run_stage3_from(start_t=t) 的语义完全一致。
    """
    device = state.device
    bsize = state.shape[0]
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    half_dt = 0.5 / num_steps  # 用于时间步匹配的容差

    x_t = noise
    timestep = torch.tensor(1.0, dtype=torch.float32, device=device)
    intermediates = {}

    while timestep >= -dt / 2:
        t_val = timestep.item()
        # 检查是否需要在此时间步保存（使用半步容差避免浮点误差）
        for st in save_timesteps:
            if abs(t_val - st) < half_dt:
                intermediates[st] = x_t.clone()
                break

        expanded_time = timestep.expand(bsize)
        v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
        x_t = x_t + dt * v_t
        timestep += dt

    return x_t, intermediates
```

---

## 4. 各 Stage 的张量形状总结

基于代码分析得出的形状，以 `B=1`，`gemma_2b`（width=2048）为例：

### Stage 1 输出

| 字段 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `state` | `[B, 32]` | float32 / bfloat16 | action_dim=32 |
| `prefix_embs` | `[B, prefix_len, 2048]` | bfloat16 | Pi0: prefix_len≈48+视觉token; Pi0.5: ≈200+视觉token |
| `prefix_pad_masks` | `[B, prefix_len]` | bool | |
| `prefix_att_2d_masks_4d` | `[B, 1, prefix_len, prefix_len]` | float32 | additive mask，0.0 或 -inf |
| `prefix_position_ids` | `[B, prefix_len]` | int64 | |

> **注意**：SigLIP 图像 token 数取决于图像数量：3 张图 × 256 token/图 = 768 视觉 token。
> prefix_len（Pi0）≈ 816（768 视觉 + 48 语言）；Pi0.5 ≈ 968（768 + 200）。

### Stage 2 输出

| 字段 | 类型 | 说明 |
|------|------|------|
| `past_key_values` | `DynamicCache` | Gemma 2B 的 KV cache，18 层，每层 k/v 各一个张量 |
| `stage1` | `Stage1Output` | 完整 Stage 1 输出的引用（不额外拷贝） |

`past_key_values` 形状参考（per layer）：
- key: `[B, num_kv_heads, prefix_len, head_dim]`
- value: `[B, num_kv_heads, prefix_len, head_dim]`

### Stage 3 输出

| 字段 | 形状 | dtype | 说明 |
|------|------|-------|------|
| `action_chunk` | `[B, 50, 32]` | float32 | action_horizon=50, action_dim=32 |
| `intermediates[t]` | `[B, 50, 32]` | float32 | 可选，与 action_chunk 同形 |

---

## 5. 注意事项与风险

### 5.1 `past_key_values` 的生命周期

`DynamicCache` 对象由 HuggingFace Transformers 维护，内部含 GPU tensor。
- **禁止** `deepcopy` 或 `clone`（会触发大量 VRAM 分配）
- `Stage2Output` 持有其引用，缓存系统若要存储 KV cache 需要另行设计（当前 Step 1-4 无需存储 KV cache）
- 在 `denoise_step()` 调用中，`past_key_values` 只读（use_cache=False），安全共享

### 5.2 `torch.compile` 兼容性

```python
# pi0_pytorch.py __init__:
if config.pytorch_compile_mode is not None:
    self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)
```

- 新增的 `run_stage*` 方法**不会**被 compile（只有 `sample_actions` 被 compile）
- 这是预期行为：缓存系统的拦截器路径不走 compiled `sample_actions`
- `run_stage3(return_intermediates=False)` 内部调用 `_stage3_action_expert()`，该方法同样不被 compile，性能等价于未 compile 的路径（与缓存命中无关的 baseline 用 `sample_actions` 的 compiled 版本）

### 5.3 Pi0.5 的 `state` 在 Stage 1 中的角色

Pi0.5 的 `state` 在变换管线（`transforms.py`）中已被离散化并合并进 `lang_tokens`。
`Stage1Output.state` 存放的是**原始连续状态向量**，在 Stage 3 的 `embed_suffix` 中对 Pi0.5 **不会被 `state_proj` 使用**（Pi0.5 没有 `state_proj`）。
该字段的存在是为了让缓存键构建器（`QueryKeyBuilder`）访问连续状态向量进行相似度计算。

### 5.4 数值一致性要求

Step 1 验证的核心要求：

```python
original = model.sample_actions(device, observation)
stage2 = model.run_stage2(model.run_stage1(observation))
new = model.run_stage3(stage2).action_chunk

assert torch.allclose(original, new, atol=1e-5), "数值不一致"
```

由于 `run_stage3(return_intermediates=False)` 直接调用 `_stage3_action_expert()`，两条路径在计算图层面**完全相同**，应精确相等（atol=0）。

### 5.5 现有计时系统不受影响

`policy.py` 中的 `_staged_inference` 路径调用的是 `_stage1_token_prep` 等私有方法，
与新增的 `run_stage1` 等公共方法**完全独立**。Step 1 之后 `policy.py` 无需任何修改。

---

## 6. 不需要做的事（防止过度设计）

- **不需要**修改 `_stage1_token_prep` / `_stage2_llm_backbone` / `_stage3_action_expert`（只包装，不修改）
- **不需要**修改 `sample_actions()`（保持 compiled 路径不变）
- **不需要**修改 `policy.py`（计时系统保持现状）
- **不需要**实现 `CacheOrchestrator` 或 `InferenceInterceptor`（Step 4 的工作）
- **不需要**实现 `VectorStore`（Step 3 的工作）
- **不需要**对 Pi0.5 实现自回归子任务生成（当前 PyTorch 实现未支持）

---

## 7. 开发阶段划分

### 阶段 A：Plan（本文档）

- [x] 阅读 `pi0_pytorch.py` 全文，标注各阶段边界
- [x] 阅读 `policy.py`，确认现有计时系统结构
- [x] 理解 Pi0 / Pi0.5 差异
- [x] 确认 `past_key_values` 结构和使用方式
- [x] 完成本规划文档

### 阶段 B：Coding

**B1：添加数据类**
- 在 `pi0_pytorch.py` 顶部（`class PI0Pytorch` 之前）添加 `Stage1Output`、`Stage2Output`、`Stage3Output`
- 引入 `dataclasses.dataclass`、`typing.Optional`、`typing.Any`

**B2：添加 `_stage3_with_intermediates` 私有辅助方法**
- 实现与 `_stage3_action_expert` 等价的循环，加入中间状态保存逻辑

**B3：添加 `run_stage1` / `run_stage2` / `run_stage3` / `run_stage3_from` 公共方法**
- 实现上述接口设计中的四个方法

### 阶段 C：验证

**C1：数值一致性验证**
```python
# 使用 debug_pi05 config 或 debug config
model = PI0Pytorch(config)
# 随机输入
obs = create_dummy_observation(config)

with torch.no_grad():
    original = model.sample_actions(device, obs)
    stage1 = model.run_stage1(obs)
    stage2 = model.run_stage2(stage1)
    stage3 = model.run_stage3(stage2)
    new = stage3.action_chunk

diff = (original - new).abs().max().item()
assert diff == 0.0, f"数值差异: {diff}"  # 应精确相等
```

**C2：Warm Start 路径验证**
```python
# return_intermediates=True 时，中间状态保存是否正确
stage3_with = model.run_stage3(stage2, return_intermediates=True)
assert 0.5 in stage3_with.intermediates
assert stage3_with.intermediates[0.5].shape == (B, 50, 32)

# run_stage3_from 从 t=0.0 开始（仅做 0 步，等价于直接返回 x_0 近似）
# 以及从 t=1.0 开始（等价于完整去噪）的边界情况
```

**C3：Pi0 与 Pi0.5 分别验证**
- 用 `debug` config（Pi0）和 `debug_pi05` config（Pi0.5）分别运行 C1

**C4：policy.py 集成验证**
- 确认 `policy.py` 的 `staged_inference` 路径（使用私有方法）在 Step 1 修改后仍然正常输出 `stage_timing`
- 即：新增代码不影响已有 `_stage1_token_prep` 等私有方法的行为

---

## 8. 文件修改摘要

```
src/openpi/models_pytorch/pi0_pytorch.py
  + import dataclasses, Optional, Any (如未导入)
  + Stage1Output dataclass
  + Stage2Output dataclass
  + Stage3Output dataclass
  + PI0Pytorch.run_stage1()
  + PI0Pytorch.run_stage2()
  + PI0Pytorch.run_stage3()
  + PI0Pytorch.run_stage3_from()
  + PI0Pytorch._stage3_with_intermediates()
  (所有已有代码不修改)
```

总计：约 +120 行，零修改行。
