# Pi0.5 模型 Forward / Inference 逻辑分析

> 代码仓库：`/home/ziyang10/openpi`
> 主要分析文件：
> - JAX 实现：`src/openpi/models/pi0.py`
> - PyTorch 实现：`src/openpi/models_pytorch/pi0_pytorch.py`
> - 模型配置：`src/openpi/models/pi0_config.py`
> - Gemma 骨干网络：`src/openpi/models/gemma.py` / `src/openpi/models_pytorch/gemma_pytorch.py`
> - 状态离散化 Tokenizer：`src/openpi/models/tokenizer.py`

---

## 一、模型整体架构

Pi0.5 是一个基于 **Flow Matching** 的 VLA（Vision-Language-Action）模型，核心骨干是 **PaliGemma**（2B 参数）+ **Action Expert**（Gemma 300M），两者共享注意力层但权重独立。

Inference 阶段采用 **Euler ODE Solver**，从高斯噪声 `x₁` 迭代去噪，最终得到预测动作 `x₀`。

```
输入: 图像 + 语言提示(含离散化 state) + 高斯噪声 x_t(t=1.0)
         │
         ├─ [只算一次] embed_prefix → KV Cache
         │
         └─ while t ∈ [1.0 → 0.0]:
                embed_suffix(x_t, t) → adaRMS cond
                PaliGemma(suffix | prefix KV cache) → v_t
                x_t = x_t + dt * v_t   (Euler step)
                t += dt                 (dt = -1/num_steps)
         │
输出: x_0 (去噪后的预测动作，形状 [B, action_horizon, action_dim])
```

---

## 二、模型配置（Pi0Config）

**文件**：`src/openpi/models/pi0_config.py`

```python
@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: str = "gemma_2b"        # 主干 LLM：PaliGemma 2B
    action_expert_variant: str = "gemma_300m"  # Action Expert：Gemma 300M
    action_dim: int = 32                        # 动作维度（电机数量）
    action_horizon: int = 50                    # 预测动作步数
    max_token_len: int = None                   # 自动：pi0.5=200，pi0=48
    pi05: bool = False                          # True 表示使用 pi0.5 路径
    discrete_state_input: bool = None           # 自动：pi05=True 时为 True
```

与 Pi0 相比，Pi0.5 的两个核心差异由 `pi05=True` 控制：
1. Robot state 以**离散文字 token** 形式输入 prefix（而非 suffix 里的连续向量）
2. Timestep 通过 **adaRMSNorm** 注入每层（而非与 action token 拼接后走 MLP）

---

## 三、模型初始化

**文件**：`src/openpi/models/pi0.py`，第 66–103 行

```python
class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        # ...
        # Action Expert 在 pi0.5 模式下启用 adaRMS
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                adarms=config.pi05,          # pi0.5 启用 adaRMSNorm
            )
        )
        llm.lazy_init(use_adarms=[False, True] if config.pi05 else [False, False])

        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width)

        if config.pi05:
            # pi0.5：时间步单独走 MLP，输出作为 adaRMS 条件向量
            self.time_mlp_in  = nnx.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width)
        else:
            # pi0：state 用连续向量投影，action + time 拼接后走 MLP
            self.state_proj           = nnx.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in   = nnx.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out  = nnx.Linear(action_expert_config.width, action_expert_config.width)

        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim)
```

---

## 四、embed_prefix()：构建 Prefix Tokens（每次推理只算一次）

**文件**：`src/openpi/models/pi0.py`，第 105–137 行

Prefix 包含**图像 token** 和**语言 token**（Pi0.5 的 state 已在 tokenizer 阶段离散化进语言序列）。

```python
def embed_prefix(self, obs):
    tokens, input_mask, ar_mask = [], [], []

    # 1. 图像 → SigLIP 视觉编码器 → 视觉 token
    for name in obs.images:
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        tokens.append(image_tokens)
        input_mask.append(...)
        ar_mask += [False] * image_tokens.shape[1]   # 图像 token 互相关注

    # 2. 语言 token（pi0.5 中含离散 state，如 "State: 128 64 200 ..."）
    if obs.tokenized_prompt is not None:
        tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
        tokens.append(tokenized_inputs)
        ar_mask += [False] * tokenized_inputs.shape[1]  # 图像+语言全局关注

    return jnp.concatenate(tokens, axis=1), input_mask, ar_mask
```

**注意力模式**：图像和语言 token 之间是**双向全局注意力**（prefix-LM 模式，ar_mask=False）。

### State 离散化（Pi0.5 特有）

**文件**：`src/openpi/models/tokenizer.py`

Pi0.5 在 tokenizer 阶段将连续 state 向量离散化为文字 token，拼入 prompt：

```python
# 将 state 映射到 [0, 255] 的离散 bin，变为文字
discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
state_str = " ".join(map(str, discretized_state))
full_prompt = f"Task: {task_text}, State: {state_str};\nAction: "
```

这使得 `max_token_len` 从 pi0 的 48 扩大到 pi0.5 的 **200**。

---

## 五、embed_suffix()：构建 Suffix Tokens（每个去噪步都重算）

**文件**：`src/openpi/models/pi0.py`，第 139–186 行

Suffix 包含**带噪动作 token**，以及通过 adaRMSNorm 注入的 timestep 条件。

```python
def embed_suffix(self, obs, noisy_actions, timestep):
    tokens, input_mask, ar_mask = [], [], []

    # pi0 才有 state token，pi0.5 的 state 已在 prefix 里
    if not self.pi05:
        state_token = self.state_proj(obs.state)[:, None, :]
        tokens.append(state_token)
        ar_mask += [True]    # 语言/图像不能关注 state token

    # 1. 动作 → 线性投影到 action expert 宽度
    action_tokens = self.action_in_proj(noisy_actions)    # [B, action_horizon, width]

    # 2. Timestep → sincos 位置编码（对 [0,1] 区间敏感）
    time_emb = posemb_sincos(timestep, width, min_period=4e-3, max_period=4.0)

    if self.pi05:
        # pi0.5：time 走独立 MLP → adaRMS 条件向量
        time_emb = swish(self.time_mlp_in(time_emb))
        time_emb = swish(self.time_mlp_out(time_emb))
        action_expert_tokens = action_tokens    # suffix token 就是纯动作 token
        adarms_cond = time_emb                  # [B, width]，注入每层 adaRMSNorm
    else:
        # pi0：action + time 拼接后走 MLP
        time_tokens = repeat(time_emb, "b emb -> b s emb", s=action_horizon)
        action_time_tokens = concat([action_tokens, time_tokens], axis=-1)
        action_expert_tokens = mlp(action_time_tokens)
        adarms_cond = None

    tokens.append(action_expert_tokens)
    # 第一个 action token 对语言/图像不可见（causal），后续 action token 互相 causal
    ar_mask += [True] + [False] * (action_horizon - 1)

    return concat(tokens), input_mask, ar_mask, adarms_cond
```

**注意力模式**：动作 token 之间是 **causal（单向）注意力**，且语言/图像 token **不能关注**动作 token。

---

## 六、adaRMSNorm：Timestep 注入机制（Pi0.5 核心创新）

**文件**：`src/openpi/models_pytorch/gemma_pytorch.py`

Pi0.5 用 adaRMSNorm（Adaptive RMS Normalization）在 Action Expert 的每一个 Transformer 层中注入 timestep 信息：

```python
# 在 Action Expert 的每个 Transformer 层里
hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond)
# ... 注意力计算 ...
out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond)
```

`adarms_cond` 是一个 `[B, width]` 的向量（time MLP 的输出），用于**调制每层 RMSNorm 的 scale 和 gate**，使 timestep 信息深入渗透进网络的每一层，而非只在输入 token 层面拼接一次。

---

## 七、compute_loss()：训练时的 Forward Pass

**文件**：`src/openpi/models/pi0.py`，第 188–214 行

训练时 prefix + suffix 一次性输入，采用 flow matching 目标函数：

```python
def compute_loss(self, rng, observation, actions, *, train=False):
    # Flow matching：在 noise 和 clean action 之间插值
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    x_t = time * noise + (1 - time) * actions    # 插值后的带噪动作
    u_t = noise - actions                          # 目标速度场

    # 一次性 forward（prefix + suffix 同时输入，无 KV cache）
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

    attn_mask = make_attn_mask(concat([prefix_mask, suffix_mask]),
                               concat([prefix_ar_mask, suffix_ar_mask]))
    positions = cumsum(input_mask) - 1

    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, adarms_cond]    # prefix 侧普通 RMSNorm，suffix 侧 adaRMSNorm
    )

    # 取最后 action_horizon 个 token 预测速度场
    v_t = self.action_out_proj(suffix_out[:, -action_horizon:])

    # MSE loss：让预测速度场逼近真实速度场
    return mean(square(v_t - u_t), axis=-1)
```

时间步 `t` 服从 **Beta(1.5, 1)** 分布（偏向 t=1，即噪声端），范围截断在 [0.001, 1.0]。

---

## 八、sample_actions()：Inference 完整流程

**文件**：`src/openpi/models/pi0.py`，第 216–279 行

### 8.1 建立 Prefix KV Cache（只算一次）

```python
def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
    dt = -1.0 / num_steps    # 默认步数 10，dt = -0.1

    # 初始化纯高斯噪声
    noise = jax.random.normal(rng, (batch_size, action_horizon, action_dim))

    # 只跑一次 prefix forward，缓存 K/V（图像+语言 token 对应的 K/V）
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = cumsum(prefix_mask) - 1
    _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
```

### 8.2 Euler 去噪循环

```python
    def step(carry):
        x_t, time = carry

        # 每步重新 embed suffix（因为 x_t 和 time 都在变）
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, broadcast_to(time, batch_size)
        )

        # 构建 suffix 关注 prefix 的 mask
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = repeat(prefix_mask, "b p -> b s p", s=suffix_len)
        full_attn_mask = concat([prefix_attn_mask, suffix_attn_mask], axis=-1)
        # full_attn_mask shape: [B, suffix_len, prefix_len + suffix_len]

        positions = sum(prefix_mask)[:, None] + cumsum(suffix_mask) - 1

        # suffix 通过 KV cache 关注 prefix（图像+语言），同时 causal 关注自身
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],      # prefix 侧直接用 KV cache，不重算
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond]
        )

        # 预测速度场
        v_t = self.action_out_proj(suffix_out[:, -action_horizon:])

        # Euler step：沿速度场方向前进
        return x_t + dt * v_t, time + dt

    def cond(carry):
        x_t, time = carry
        return time >= -dt / 2    # 浮点容差

    x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
    return x_0
```

---

## 九、PyTorch 版本对应关系

**文件**：`src/openpi/models_pytorch/pi0_pytorch.py`

PyTorch 版本与 JAX 版本逻辑完全一致，以下是关键方法对应关系：

| JAX 方法 | PyTorch 方法 | 位置 |
|---|---|---|
| `embed_prefix()` | `embed_prefix()` | 第 187 行 |
| `embed_suffix()` | `embed_suffix()` | 第 238 行 |
| `compute_loss()` | `forward()` | 第 317 行 |
| `sample_actions()` | `sample_actions()` | 第 377 行 |
| 循环中的 `step()` | `denoise_step()` | 第 422 行 |

PyTorch 版本将每次去噪步单独封装成 `denoise_step()`：

```python
# src/openpi/models_pytorch/pi0_pytorch.py，第 407–419 行
x_t = noise
time = torch.tensor(1.0, ...)
while time >= -dt / 2:
    v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time)
    x_t = x_t + dt * v_t
    time += dt
return x_t
```

---

## 十、Pi0 vs Pi0.5 核心差异汇总

| 方面 | Pi0 | Pi0.5 |
|---|---|---|
| **State 输入方式** | suffix 里 1 个连续 state token（linear proj） | prefix 里离散化文字 token（256 bins） |
| **Timestep 注入** | `[action ‖ time_repeated]` → MLP → token | time → MLP → adaRMSNorm 调制每层 |
| **Suffix 构成** | state(1) + action(50) = 51 tokens | action(50) = 50 tokens |
| **`max_token_len`** | 48 | 200（state 变文字，prefix 更长）|
| **归一化层** | 普通 RMSNorm | Action Expert 用 adaRMSNorm |
| **`pi05` 标志** | `False` | `True` |

**配置文件**：`src/openpi/models/pi0_config.py`

---

## 十一、注意力结构总结

```
Prefix (图像 + 语言 + 离散 state)          Suffix (动作 tokens)
┌──────────────────────────────┐         ┌────────────────────────┐
│ img_tok ... lang_tok ...     │ ←←←←←← │ act_0  act_1 ... act_49│
│  双向全局注意力 (ar=False)    │         │   causal 注意力        │
│  PaliGemma 2B 处理           │         │  Action Expert 300M     │
│                              │  ×      │  + adaRMSNorm(time_emb) │
└──────────────────────────────┘         └────────────────────────┘
        ↑ KV Cache（推理时只算一次）              ↑ 每步重算

注：动作 token 可以关注图像/语言（通过 KV cache），但图像/语言 token 不能关注动作 token。
```

---

## 附：训练时使用的 Named Config

**文件**：`src/openpi/training/config.py`

Pi0.5 相关的训练配置（均设置 `pi05=True`）：
- `pi05_aloha`
- `pi05_droid`
- `pi05_libero`
- `pi05_aloha_pen_uncap`
- `pi05_full_droid_finetune`
- `pi05_droid_finetune`
- `debug_pi05`

Base checkpoint：`gs://openpi-assets/checkpoints/pi05_base/params`
