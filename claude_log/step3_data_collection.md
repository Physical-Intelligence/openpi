# Step 3 Data Collection — Implementation Plan (v3, Pure Observer)

> 目标：在 PyTorch 推理服务器上收集每个 episode 的完整模型中间嵌入，按请求顺序写入 HDF5 文件，供后续缓存系统离线分析使用。
>
> **核心原则：对现有代码零侵入。`policy.infer()` 是唯一执行路径；数据收集是纯旁路 observer，关闭时真正的零开销。**

---

## 一、需求清单（每次 infer 收集）

| 数据项 | 来源 | 形状（batch=1） | dtype | 说明 |
|--------|------|-----------------|-------|------|
| `vision_0/1/2` | SigLIP `multi_modal_projector` forward hook 输出 | `[256, 2048]` | float16 | 每张图单独，按 embed_prefix 循环顺序 |
| `prompt_emb` | `embed_tokens` forward hook + sqrt 缩放 | `[num_lang_tokens, 2048]` | float16 | Pi0.5 含离散化 state token |
| `robot_state` | `policy.infer()` 返回 dict 的 `"state"` 字段 | `[32]` | float32 | 归一化后的连续 state，直接从返回值取 |
| `noise_action_1`…`(N-1)` | `action_in_proj` forward hook，跳过第 1 次调用（纯高斯），取第 2-N 次 | `[50, 32]` | float32 | 去噪步骤 1-(N-1) 的 x_t，N 由实际触发次数动态推断 |
| `clean_action` | hook 计算：`action_in_captures[-1] + dt × action_out_captures[-1]`，dt = -1/N | `[50, 32]` | float32 | 归一化空间内的 x_0，dt 动态推断 |

---

## 二、核心设计：纯 Observer

### 2.1 关键约束确认（代码验证结论）

| 约束 | 验证结论 |
|------|---------|
| `action_in_proj` / `action_out_proj` 是 `PI0Pytorch` 的直接属性 | ✓ `pi0_pytorch.py:165-166` |
| 两者在推理路径中直接调用（非 compile wrapper 覆盖） | ✓ 推理时 `_apply_checkpoint` 只在 `self.training` 时用 checkpoint，否则直接调用 |
| `embed_tokens` 只在 Stage 1 触发，Stage 2/3 传入 `inputs_embeds` 跳过 | ✓ `_stage2_llm_backbone:453` 和 `denoise_step:707` 均传 `inputs_embeds`，HF 跳过 embed |
| `_stage3_action_expert`（推理版，line 462）调用 `denoise_step()` 10 次 | ✓ 每次 denoise_step 触发 action_in_proj 和 action_out_proj 各一次 |
| `policy.infer()` 返回 dict 含 `"state"` 字段 | ✓ Policy:infer 和 InferenceInterceptor:infer 均返回 `{"state":..., "actions":...}` |

### 2.2 四个 Hook 目标（挂在 `PI0Pytorch` 实例上）

| Hook | 挂载目标路径 | 触发时机 | 捕获内容 |
|------|------------|---------|---------|
| `_vision_hook` | `model.paligemma_with_expert.paligemma.multi_modal_projector` | Stage 1，embed_prefix 循环内每张图一次 | `out`：`[B, 256, 2048]` |
| `_lang_hook` | `model.paligemma_with_expert.paligemma.language_model.embed_tokens` | Stage 1，embed_language_tokens 一次；Stage 2/3 不触发 | `out * sqrt(dim)`：`[B, N_lang, 2048]` |
| `_action_in_hook` | `model.action_in_proj` | Stage 3 每步去噪一次，共 10 次 | `inp[0]`（x_t 输入）：`[B, 50, 32]` |
| `_action_out_hook` | `model.action_out_proj` | Stage 3 每步去噪一次，共 10 次 | `out`（v_t 输出）：`[B, 50, 32]` |

### 2.3 x_t / v_t 对应关系

```
去噪步序  t          action_in_proj 触发   action_out_proj 触发   x_t 更新
step 1   1.0        call[0] → x_noise    call[0] → v_1          x_1 = x_noise + dt*v_1
step 2   1-1/N      call[1] → x_1        call[1] → v_2          x_2 = x_1 + dt*v_2
...
step N   1/N        call[N-1] → x_{N-1}  call[N-1] → v_N        x_0 = x_{N-1} + dt*v_N

N = len(action_in_captures)     （从实际 hook 触发次数动态推断，默认 10）
dt = -1.0 / N                   （动态计算，不硬编码）

noise_action_i  = action_in_captures[i]   i=1..N-1  （跳过 [0] = pure Gaussian）
clean_action    = action_in_captures[-1] + dt * action_out_captures[-1]
```

### 2.4 CollectionPolicy 行为

```
当 --collect 关闭 或 collect 未包装时：
    CollectionPolicy 不存在，policy.infer() 照常执行，零影响

当 --collect 开启，且 _collecting=False（两个 episode 之间）：
    CollectionPolicy.infer(obs, *, noise=None)
      → return self._policy.infer(obs, noise=noise)
    纯委托，无 hook，无额外内存分配

当 --collect 开启，且 _collecting=True（episode 进行中）：
    CollectionPolicy.infer(obs, *, noise=None)
      1. 注册 4 个临时 forward hook
      2. result = self._policy.infer(obs, noise=noise)   ← 唯一执行路径
      3. finally: 移除所有 hook（保证异常也移除）
      4. 从 captures 构建 InferenceEmbeddings
      5. self._collector.record_inference(embs)
      6. return result
```

### 2.5 Wrapper 顺序（serve_policy.py 中）

```python
policy = create_policy(args)               # 原始 Policy（PyTorch）

# 第一步：内层 wrapper（需要直接访问 Policy._model 等私有字段）
if args.cache:
    policy = InferenceInterceptor(policy, timer=timer)
if args.record:
    policy = PolicyRecorder(policy, "policy_records")

# 第二步：CollectionPolicy 作为最外层 wrapper
# 它只调用 self._policy.infer()，不关心 self._policy 的具体类型
if args.collect:
    policy = CollectionPolicy(policy, collector)
```

---

## 三、新建文件

### 3.1 `src/openpi/collect/__init__.py`

空文件。

---

### 3.2 `src/openpi/collect/data_collector.py`

```python
"""EpisodeDataCollector: 线程安全的 episode 级 HDF5 写入器。

按 episode 批量缓存 InferenceEmbeddings，episode 结束时原子写入 HDF5。
即使 episode 中途断连，on_task_end 也会触发 flush（保数据不丢失）。
"""
from __future__ import annotations

import datetime
import logging
import pathlib
from dataclasses import dataclass, field

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceEmbeddings:
    """单次 infer 的采集结果。所有数组均已 squeeze 掉 batch dim。"""
    vision_embs: list[np.ndarray]         # len = num_images, 每个 [num_img_tokens, emb_dim], float16
    prompt_emb: np.ndarray                # [num_lang_tokens, emb_dim], float16
    robot_state: np.ndarray               # [action_dim], float32
    noise_action_steps: list[np.ndarray]  # len=9, 每个 [action_horizon, action_dim], float32
    clean_action: np.ndarray              # [action_horizon, action_dim], float32


class EpisodeDataCollector:
    """缓存单个 episode 的所有 InferenceEmbeddings，episode 结束时写入 HDF5。"""

    def __init__(self, base_dir: str) -> None:
        self._base_dir = pathlib.Path(base_dir)
        self._buffer: list[InferenceEmbeddings] = []
        self._experiment = "unknown"
        self._task = ""
        self._episode_id = -1

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def on_episode_start(self, experiment: str, task: str, episode_id: int) -> None:
        """新 episode 开始，清空缓冲区。"""
        self._buffer = []
        self._experiment = experiment
        self._task = task
        self._episode_id = episode_id
        logger.info("EpisodeDataCollector: episode %d started (%s / %s)", episode_id, experiment, task)

    def record_inference(self, embs: InferenceEmbeddings) -> None:
        """将单次 infer 数据追加到当前 episode 缓冲区。"""
        self._buffer.append(embs)

    def on_episode_end(self, success: bool) -> None:
        """Episode 结束，将缓冲区原子写入 HDF5。"""
        if not self._buffer:
            logger.warning("EpisodeDataCollector: episode %d has no data, skipping write.", self._episode_id)
            return

        out_dir = self._base_dir / self._experiment
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = out_dir / f"episode_{self._episode_id:04d}_{ts}.h5"
        tmp_path = path.with_suffix(".h5.tmp")

        try:
            with h5py.File(tmp_path, "w") as f:
                f.attrs["experiment_name"] = self._experiment
                f.attrs["task"] = self._task
                f.attrs["episode_id"] = self._episode_id
                f.attrs["num_steps"] = len(self._buffer)
                f.attrs["timestamp"] = datetime.datetime.now().isoformat()
                f.attrs["success"] = success

                for step_idx, embs in enumerate(self._buffer):
                    grp = f.create_group(f"step_{step_idx:04d}")
                    for i, v in enumerate(embs.vision_embs):
                        grp.create_dataset(f"vision_{i}", data=v, compression="lzf")
                    grp.create_dataset("prompt_emb", data=embs.prompt_emb, compression="lzf")
                    grp.create_dataset("robot_state", data=embs.robot_state)
                    for i, ns in enumerate(embs.noise_action_steps, start=1):
                        grp.create_dataset(f"noise_action_{i}", data=ns)
                    grp.create_dataset("clean_action", data=embs.clean_action)

            tmp_path.rename(path)
            logger.info(
                "EpisodeDataCollector: episode %d written → %s (%d steps, success=%s)",
                self._episode_id, path, len(self._buffer), success,
            )
        except Exception:
            logger.exception("EpisodeDataCollector: failed to write episode %d", self._episode_id)
            tmp_path.unlink(missing_ok=True)
        finally:
            self._buffer = []
```

---

### 3.3 `src/openpi/collect/collection_policy.py`

```python
"""CollectionPolicy: 纯 observer 式数据收集包装器。

设计原则
--------
* 当 _collecting=False 时，infer() 是纯委托：return self._policy.infer(obs[, noise=noise])
  没有任何 hook、没有任何额外计算。
* 当 _collecting=True 时，注册 4 个临时 forward hook → 调用 self._policy.infer()
  （唯一执行路径）→ finally 块移除所有 hook → 从 captures 构建 InferenceEmbeddings。
* 对现有所有代码（Policy、InferenceInterceptor、PolicyRecorder、PI0Pytorch
  及其所有方法）零修改、零侵入。

Wrapper 顺序（serve_policy.py 中）
--------------------------------------
  policy = create_policy(args)
  if args.cache:  policy = InferenceInterceptor(policy, ...)   # 内层
  if args.record: policy = PolicyRecorder(policy, ...)          # 内层
  if args.collect: policy = CollectionPolicy(policy, collector) # 最外层

TaskLifecycle 转发
------------------
  on_task_begin / on_task_end 通过 __getattr__ 自动转发到 self._policy。
  on_task_end 在转发前先 flush 未完成 episode（防断连丢数据）。
  on_episode_start / on_episode_end 由 WebsocketPolicyServer 经带内控制消息触发。

noise kwarg 兼容性
------------------
  PolicyRecorder.infer(self, obs) 不接受 noise 参数。为避免 TypeError，
  CollectionPolicy 只在 noise is not None 时才传 noise kwarg。
  在 WebSocket serving 场景，客户端从不发送 noise，noise 恒为 None，
  PolicyRecorder 完全不受影响。

num_steps / dt 动态推断
-----------------------
  不硬编码步数。infer() 结束后从 len(action_in_captures) 动态计算
  N = num_steps 和 dt = -1/N，自动适配任意 sample_kwargs 中的 num_steps。
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
from openpi_client import base_policy as _base_policy

from openpi.collect.data_collector import EpisodeDataCollector, InferenceEmbeddings

logger = logging.getLogger(__name__)
# 注意：不在模块级硬编码 dt / num_steps。dt 在 _record() 内从 hook 触发次数动态计算。


def _find_inner_model(policy: Any) -> Any:
    """遍历 wrapper 链，找到第一个暴露 _model 属性的对象并返回其 _model。

    支持：Policy → _model = PI0Pytorch
          InferenceInterceptor → _model = PI0Pytorch（借用自 Policy）
    """
    obj = policy
    while obj is not None:
        if hasattr(obj, "_model"):
            return obj._model
        obj = getattr(obj, "_policy", None)
    raise ValueError(
        "CollectionPolicy: cannot find _model in policy chain. "
        "Only PyTorch policies (PI0Pytorch) are supported."
    )


class CollectionPolicy(_base_policy.BasePolicy):
    """透明 observer 包装器，开启时收集 infer 中间嵌入。

    Args:
        policy: 任意实现了 BasePolicy.infer() 的对象。可以是 Policy、
                InferenceInterceptor、PolicyRecorder，或任意嵌套组合。
        collector: EpisodeDataCollector 实例。
    """

    def __init__(self, policy: Any, collector: EpisodeDataCollector) -> None:
        self._policy = policy
        self._collector = collector
        self._collecting = False

        # 访问底层 PI0Pytorch 模型以注册 hook
        self._inner_model = _find_inner_model(policy)

    # ------------------------------------------------------------------
    # BasePolicy interface
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata

    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        """推理入口。

        不收集时：纯委托，零开销。
        收集时：注册临时 hook，经 self._policy.infer() 触发，finally 移除 hook。

        noise kwarg 说明：PolicyRecorder.infer(self, obs) 不接受 noise 参数。
        为避免 TypeError，noise=None 时不传 kwarg（空 **{}），
        noise is not None 时才透传。在 WebSocket serving 场景 noise 恒为 None。
        """
        # noise=None 时不传 kwarg，避免 PolicyRecorder 等不接受 noise 的内层 TypeError
        _infer_kwargs = {"noise": noise} if noise is not None else {}

        if not self._collecting:
            # --- 纯委托路径：无 hook，无额外计算 ---
            return self._policy.infer(obs, **_infer_kwargs)

        # --- 收集路径：注册 4 个临时 hook ---
        vision_captures: list[torch.Tensor] = []
        lang_capture: list[torch.Tensor | None] = [None]
        action_in_captures: list[torch.Tensor] = []
        action_out_captures: list[torch.Tensor] = []

        def _vision_hook(module, inp, out):
            # multi_modal_projector: 每张图触发一次，out 形状 [B, 256, emb_dim]
            vision_captures.append(out.detach().clone())

        def _lang_hook(module, inp, out):
            # embed_tokens: Stage 1 触发一次，out 形状 [B, N_lang, emb_dim]
            # embed_prefix 内部会 * sqrt(dim)，此处同步补上，与 embed_prefix 输出一致
            lang_capture[0] = (out * math.sqrt(out.shape[-1])).detach().clone()

        def _action_in_hook(module, inp, out):
            # action_in_proj: Stage 3 每个去噪步触发一次，inp[0] = x_t，形状 [B, 50, 32]
            action_in_captures.append(inp[0].detach().clone())

        def _action_out_hook(module, inp, out):
            # action_out_proj: Stage 3 每个去噪步触发一次，out = v_t，形状 [B, 50, 32]
            action_out_captures.append(out.detach().clone())

        m = self._inner_model
        handles = [
            m.paligemma_with_expert.paligemma.multi_modal_projector
             .register_forward_hook(_vision_hook),
            m.paligemma_with_expert.paligemma.language_model.embed_tokens
             .register_forward_hook(_lang_hook),
            m.action_in_proj.register_forward_hook(_action_in_hook),
            m.action_out_proj.register_forward_hook(_action_out_hook),
        ]

        try:
            result = self._policy.infer(obs, **_infer_kwargs)
        finally:
            for h in handles:
                h.remove()

        # --- 后处理：构建 InferenceEmbeddings ---
        try:
            self._record(result, vision_captures, lang_capture[0], action_in_captures, action_out_captures)
        except Exception:
            logger.exception("CollectionPolicy: failed to record inference embeddings; skipping step.")

        return result

    # ------------------------------------------------------------------
    # Episode lifecycle（由 WebsocketPolicyServer 经带内控制消息调用）
    # ------------------------------------------------------------------

    def on_episode_start(self, experiment: str, task: str, episode_id: int) -> None:
        self._collector.on_episode_start(experiment, task, episode_id)
        self._collecting = True

    def on_episode_end(self, success: bool) -> None:
        self._collector.on_episode_end(success)
        self._collecting = False

    # ------------------------------------------------------------------
    # TaskLifecycle 转发（由 WebsocketPolicyServer 在连接开/关时调用）
    # ------------------------------------------------------------------

    def on_task_begin(self) -> None:
        if hasattr(self._policy, "on_task_begin"):
            self._policy.on_task_begin()

    def on_task_end(self) -> None:
        # 断连时 flush 未完成的 episode，防止数据丢失
        if self._collecting and self._collector._buffer:
            logger.warning("CollectionPolicy: connection closed mid-episode, flushing partial data.")
            self._collector.on_episode_end(success=False)
            self._collecting = False
        if hasattr(self._policy, "on_task_end"):
            self._policy.on_task_end()

    def __getattr__(self, name: str) -> Any:
        """转发其余属性访问到内层 policy（如 on_task_begin/end 等未显式定义的接口）。"""
        return getattr(self._policy, name)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _record(
        self,
        result: dict,
        vision_captures: list[torch.Tensor],
        lang_emb: torch.Tensor | None,
        action_in_captures: list[torch.Tensor],
        action_out_captures: list[torch.Tensor],
    ) -> None:
        """从 hook captures 构建 InferenceEmbeddings 并提交给 collector。"""

        # ---- vision ----
        # 每个 [B=1, 256, emb_dim] → squeeze → [256, emb_dim]，float16
        vision_embs = [
            v.squeeze(0).cpu().to(torch.float16).numpy()
            for v in vision_captures
        ]

        # ---- prompt_emb ----
        # [B=1, N_lang, emb_dim] → squeeze → [N_lang, emb_dim]，float16
        if lang_emb is None:
            raise RuntimeError("CollectionPolicy: lang_emb hook did not fire; "
                               "check that the wrapped policy is a PyTorch model.")
        prompt_emb_np = lang_emb.squeeze(0).cpu().to(torch.float16).numpy()

        # ---- robot_state ----
        # policy.infer() 返回 dict 含 "state"（归一化后的连续 state，output_transform 不修改它）
        robot_state_np = np.asarray(result["state"], dtype=np.float32).flatten()

        # ---- noise_action_1..(N-1) 和 clean_action ----
        # 从实际 hook 触发次数动态推断 N 和 dt，自动适配任意 num_steps。
        # action_in_captures: N 次调用，[0]=纯高斯（跳过），[1..N-1]=noise_action_1..(N-1)
        # action_out_captures: N 次调用，[-1]=v_N（最后一步速度）
        num_steps = len(action_in_captures)
        if num_steps < 2 or len(action_out_captures) != num_steps:
            raise RuntimeError(
                f"CollectionPolicy: action_in_proj fired {num_steps} times, "
                f"action_out_proj fired {len(action_out_captures)} times. "
                "Expected equal counts ≥ 2. Check Stage 3 execution."
            )
        dt = -1.0 / num_steps   # 动态计算，不硬编码 -0.1

        noise_action_steps = [
            action_in_captures[i].squeeze(0).cpu().numpy().astype(np.float32)
            for i in range(1, num_steps)  # 跳过 [0]（纯高斯）
        ]

        # ---- clean_action ----
        # x_0 = x_{N-1} + dt * v_N
        # x_{N-1} = action_in_captures[-1]（第 N 次 action_in_proj 输入）
        # v_N     = action_out_captures[-1]（第 N 次 action_out_proj 输出）
        x_last = action_in_captures[-1].squeeze(0).cpu().float()
        v_last = action_out_captures[-1].squeeze(0).cpu().float()
        clean_action_np = (x_last + dt * v_last).numpy().astype(np.float32)

        embs = InferenceEmbeddings(
            vision_embs=vision_embs,
            prompt_emb=prompt_emb_np,
            robot_state=robot_state_np,
            noise_action_steps=noise_action_steps,
            clean_action=clean_action_np,
        )
        self._collector.record_inference(embs)
```

---

## 四、修改现有文件（最小化）

### 4.1 `scripts/serve_policy.py` — 新增两个参数

在 `Args` dataclass 中添加：

```python
# 是否开启数据收集（不影响任何现有推理路径）
collect: bool = False
# 收集数据的保存根目录
collect_dir: str = "/data"
```

在 `main()` 中，**cache/record 包装之后**，添加：

```python
# CollectionPolicy 作为最外层 wrapper（最后包装）
if args.collect:
    from openpi.collect.data_collector import EpisodeDataCollector
    from openpi.collect.collection_policy import CollectionPolicy
    collector = EpisodeDataCollector(base_dir=args.collect_dir)
    policy = CollectionPolicy(policy, collector)
    logging.info("Data collection enabled → %s", args.collect_dir)
```

---

### 4.2 `src/openpi/serving/websocket_policy_server.py` — 带内控制消息识别

在 `_handler()` 的 while 循环内，`obs = msgpack_numpy.unpackb(await websocket.recv())` 之后、`self._policy.infer(obs)` 之前，插入：

```python
# 带内控制消息：episode_start / episode_end
# 由支持数据收集的客户端发送，无 __ctrl__ 键时忽略（普通推理请求）
if "__ctrl__" in obs:
    ctrl = obs["__ctrl__"]
    # 注意：ack 无条件发送，不管 policy 是否实现 on_episode_start/end。
    # 这保证了"未开启 --collect 的服务器"上客户端也不会挂死等待 ack。
    if ctrl == "episode_start":
        if hasattr(self._policy, "on_episode_start"):
            self._policy.on_episode_start(
                obs.get("__experiment__", "unknown"),
                obs.get("__task__", ""),
                obs.get("__episode_id__", -1),
            )
        await websocket.send(packer.pack({"__ack__": "episode_start"}))
    elif ctrl == "episode_end":
        if hasattr(self._policy, "on_episode_end"):
            self._policy.on_episode_end(obs.get("__success__", False))
        await websocket.send(packer.pack({"__ack__": "episode_end"}))
    continue  # 控制消息不走 infer 路径
```

---

### 4.3 `packages/openpi-client/src/openpi_client/websocket_client_policy.py` — 新增两个方法

```python
def episode_start(self, experiment: str, task: str = "", episode_id: int = -1) -> None:
    """通知服务器新 episode 开始（触发数据收集）。不影响无 --collect 的服务器。"""
    self._ws.send(self._packer.pack({
        "__ctrl__": "episode_start",
        "__experiment__": experiment,
        "__task__": task,
        "__episode_id__": episode_id,
    }))
    msgpack_numpy.unpackb(self._ws.recv())  # 等待 ack

def episode_end(self, success: bool = False) -> None:
    """通知服务器 episode 结束（触发 HDF5 写入）。"""
    self._ws.send(self._packer.pack({
        "__ctrl__": "episode_end",
        "__success__": success,
    }))
    msgpack_numpy.unpackb(self._ws.recv())  # 等待 ack
```

---

### 4.4 `examples/libero/main.py` — 各加两行

在每个 episode 的主循环前后插入信令调用：

```python
# episode 主循环之前
client.episode_start(
    experiment=args.task_suite_name,
    task=str(task_description),
    episode_id=episode_idx,
)

# ... 已有的 episode 主循环 ...

# episode 主循环之后（无论成功还是失败）
client.episode_end(success=done)
```

---

## 五、HDF5 文件结构

```
/data/{experiment}/episode_{N:04d}_{YYYYMMDD_HHMMSS_ffffff}.h5
├── attrs:
│   ├── experiment_name: str
│   ├── task:            str
│   ├── episode_id:      int
│   ├── num_steps:       int
│   ├── timestamp:       str (ISO 8601)
│   └── success:         bool
│
├── step_0000/
│   ├── vision_0      float16[256, 2048]   lzf 压缩
│   ├── vision_1      float16[256, 2048]   lzf 压缩
│   ├── vision_2      float16[256, 2048]   lzf 压缩
│   ├── prompt_emb    float16[N_lang, 2048]  lzf 压缩
│   ├── robot_state   float32[32]
│   ├── noise_action_1  float32[50, 32]
│   ├── noise_action_2  float32[50, 32]
│   │   ...
│   ├── noise_action_9  float32[50, 32]
│   └── clean_action    float32[50, 32]
├── step_0001/
│   └── ...
└── step_NNNN/
```

**存储估算（每次 infer）：**
- vision: 3 × 256 × 2048 × 2B ≈ 3.1 MB
- prompt_emb: ~200 × 2048 × 2B ≈ 0.8 MB（lzf 后更小）
- noise_action × 9 + clean_action: 10 × 50 × 32 × 4B ≈ 0.06 MB
- 合计约 **~3.5 MB/infer**，episode 50 步约 **~175 MB/episode**

---

## 六、完整兼容性矩阵

| 组件 | `--collect` 关闭 | `--collect` 开启，episode 外 | `--collect` 开启，episode 中 |
|------|:---:|:---:|:---:|
| `pi0_pytorch.py` 所有方法 | **零改动** | **零改动** | **零改动** |
| `policy.py` / `Policy.infer()` | **零改动** | **零改动** | **零改动** |
| `interceptor.py` / `InferenceInterceptor` | **零改动** | **零改动** | **零改动** |
| `PolicyRecorder` | **零改动** | **零改动** | **零改动** |
| JAX 推理路径 | 完全不变 | 完全不变 | 不涉及 |
| 未发控制消息的客户端 | 完全不变 | 完全不变 | 完全不变 |
| `WebsocketPolicyServer` | 无控制消息逻辑 | 控制消息→ack，其余不变 | 同左 |

---

## 七、实现顺序

1. **新建** `src/openpi/collect/__init__.py`（空文件）
2. **新建** `src/openpi/collect/data_collector.py`
3. **新建** `src/openpi/collect/collection_policy.py`
4. **修改** `src/openpi/serving/websocket_policy_server.py`（插入控制消息识别）
5. **修改** `packages/openpi-client/src/openpi_client/websocket_client_policy.py`（新增两个方法）
6. **修改** `scripts/serve_policy.py`（新增两个参数 + 最外层包装）
7. **修改** `examples/libero/main.py`（插入 episode_start/end 信令）

---

## 八、使用示例

```bash
# 服务器（开启数据收集）
uv run scripts/serve_policy.py \
    --env libero \
    --policy.config pi05_libero \
    --policy.dir checkpoints/pi05_libero/exp/10000 \
    --collect \
    --collect_dir /data/libero_experiments

# 客户端（无需改变命令行，main.py 内部已调用 episode_start/end）
uv run examples/libero/main.py \
    --host <server_ip> \
    --task_suite_name libero_spatial

# 输出：
# /data/libero_experiments/libero_spatial/episode_0000_20260330_120000_000000.h5
# /data/libero_experiments/libero_spatial/episode_0001_20260330_120100_000000.h5
# ...
```

---

## 九、审核要点检查

| 审核发现的问题（v2→v3） | 此版本解决方案 |
|----------------------|-------------|
| InferenceInterceptor 要求直接包裹 Policy | ✓ CollectionPolicy 作最外层，InferenceInterceptor 在内 |
| `--collect` ON 但 episode 未激活时有 hook 开销 | ✓ `_collecting=False` → 纯委托，无 hook |
| `infer()` 缺少 `noise` 参数 | ✓ `infer(obs, *, noise=None)` → `**_infer_kwargs` 透传 |
| `"state"` 字段从返回 dict 中丢失 | ✓ `return result`（完整转发 `self._policy.infer()` 的返回值）|
| 断连时 buffer 丢失 | ✓ `on_task_end()` 在转发前先 flush pending buffer |
| run_stage1_for_collection 修改了现有代码 | ✓ 完全删除，改用 4 个 forward hook |
| return_intermediates 改变执行路径 | ✓ 完全不使用，hook 在正常 policy.infer() 路径内触发 |
| **[新] 无 --collect 服务器上 client 挂死** | ✓ 服务器无条件发 ack，不依赖 policy 是否实现 on_episode_start/end |
| **[新] noise kwarg 与 PolicyRecorder 不兼容** | ✓ `noise=None` 时传 `**{}`（空），不触发 PolicyRecorder TypeError |
| **[新] num_steps / dt 硬编码** | ✓ 从 `len(action_in_captures)` 动态推断 N 和 dt，适配任意步数 |
