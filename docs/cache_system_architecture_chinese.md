# Pi0.5 Inference Cache System - Architecture Specification

> Version: 0.1 (Draft)
> Status: Design Phase
> Scope: PyTorch inference pipeline only (JAX path disabled)

---

## 1. System Goals

在 Pi0.5 的推理管线中引入多级缓存系统，通过复用历史计算结果来减少冗余推理，降低端到端延迟。系统设计遵循以下原则：

1. **与推理管线解耦**：Cache 系统作为外挂组件，通过 hook/interceptor 模式接入推理管线，不修改现有 inference 代码的内部逻辑。
2. **多级渐进式命中**：在推理管线的三个关键位置设置检查点，越早命中节省越多计算。
3. **硬件感知**：Vector DB 数据在 GPU/CPU 间智能分配，异步传输不阻塞推理计算。
4. **精确计时**：每个组件（检索、判定、数据传输）独立计时，支撑后续性能优化决策。
5. **递进式实现**：从单机单任务重复场景起步，逐步扩展到多任务/多机器人/分布式。

---

## 2. 推理管线三阶段模型

基于 Pi0.5 的 PyTorch 推理路径（`src/openpi/models_pytorch/pi0_pytorch.py`），将 inference 划分为三个阶段：

```
Stage 1: Token Preparation        Stage 2: LLM Backbone         Stage 3: Action Expert
+---------------------------+     +------------------------+     +---------------------------+
| SigLIP vision encoder     |     | Gemma 2B (PaliGemma)   |     | Gemma 300M + adaRMSNorm   |
| Prompt tokenization       |     | Prefix-LM attention    |     | Flow matching (10 steps)  |
| State discretization      |     | Generate low-level cmd |     | Euler ODE: x1 -> x0      |
| -> prefix tokens + KV     |     | (subtask prediction)   |     | -> action chunk [50, 32]  |
+---------------------------+     +------------------------+     +---------------------------+
            |                                |                                |
         [CP1]                            [CP2]                           [CP3]
      Cache Check 1                   Cache Check 2                   Cache Check 3
```

**各阶段计算量估算（单次推理）**：

| Stage | 主要计算 | 参数量 | 特点 |
|-------|---------|--------|------|
| 1. Token Prep | SigLIP forward + tokenize | ~400M | 单次前向，可并行 |
| 2. LLM Backbone | Gemma 2B autoregressive decode | ~2B | 自回归，序列依赖 |
| 3. Action Expert | 10x Gemma 300M forward | ~300M x10 | 迭代式，可部分跳过 |

---

## 3. 三个检查点的语义定义

### CP1: Vision 之后

- **触发时机**：Stage 1 完成，prefix tokens 和 KV cache 已生成。
- **可用信息**：vision embedding, prompt embedding, state embedding。
- **命中行为**：跳过 Stage 2 + Stage 3，直接输出缓存的 action chunk。
- **节省量**：最大（跳过 LLM 解码 + 全部 flow matching）。
- **风险**：最高——跳过了 subtask 预测，如果场景发生了微妙变化（如物体被移走），缓存的 subtask 可能不再正确。
- **适用场景**：高度重复的操作（如流水线上的同一动作）。

### CP2: LLM Backbone 之后

- **触发时机**：Stage 2 完成，low-level command（subtask text tokens）已生成。
- **可用信息**：CP1 的全部信息 + low-level command embedding。
- **命中行为（两种模式）**：
  - **Full hit**：跳过 Stage 3 全部，直接输出缓存的 action chunk。
  - **Partial hit (warm start)**：用缓存的中间状态 `x_t`（t < 1.0）作为 flow matching 起点，跳过部分去噪步骤。
- **节省量**：中等（跳过全部或部分 flow matching）。
- **风险**：中等——subtask 已由当前推理计算，缓存的 action 与当前场景的一致性更高。
- **适用场景**：相同 subtask 在相似场景下的复用。

### CP3: Action Expert 之后

- **触发时机**：Stage 3 完成，当前推理周期的 action chunk 已生成。
- **可用信息**：全部信息（vision + prompt + state + command + action chunk）。
- **命中行为**：不影响当前周期的输出。判定**下一个推理周期**是否可以跳过，直接执行 cache 中的后续 action chunk。
- **节省量**：最大（跳过完整的下一次推理）。
- **风险**：中等——依赖对未来状态的预测准确性。
- **适用场景**：连续动作序列具有时间局部性的场景（如长程物体搬运的中间阶段）。

### 检查点关系图

```
                    Inference Cycle N                          Cycle N+1
            ┌─────────┬─────────┬──────────┐          ┌──────────────────┐
            │ Stage 1 │ Stage 2 │ Stage 3  │          │ Stage 1,2,3      │
            │ Vision  │  LLM    │ FlowMatch│          │ (may be skipped) │
            └────┬────┴────┬────┴─────┬────┘          └────────┬─────────┘
                 │         │          │                         │
              [CP1]     [CP2]      [CP3]─── predict ──────> skip?
                 │         │          │
          hit: skip    hit: skip   hit: schedule
          S2+S3        S3 (full    next cycle's
                       or partial) action from cache
```

---

## 4. 系统架构

### 4.1 顶层组件图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        CacheOrchestrator                                 │
│  (controls all cache workflow, decoupled from inference)                  │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ CP1 Handler  │  │ CP2 Handler  │  │ CP3 Handler  │                   │
│  │              │  │              │  │              │   CheckpointHandler│
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         │                 │                 │                             │
│  ┌──────┴─────────────────┴─────────────────┴───────┐                   │
│  │              QueryKeyBuilder (pluggable)          │                   │
│  │  Converts stage outputs -> fixed-dim query vector │                   │
│  └──────────────────────┬───────────────────────────┘                   │
│                         │                                                │
│  ┌──────────────────────┴───────────────────────────┐                   │
│  │              GateFunction (pluggable)             │                   │
│  │  Decides: should we even search cache?            │                   │
│  │  (heuristic / lightweight model / always-on)      │                   │
│  └──────────────────────┬───────────────────────────┘                   │
│                         │                                                │
│  ┌──────────────────────┴───────────────────────────┐                   │
│  │              SimilarityJudge (pluggable)          │                   │
│  │  Given search results, decide: hit or miss?       │                   │
│  │  (threshold / learned / composite)                │                   │
│  └──────────────────────────────────────────────────┘                   │
└──────────────────────────┬───────────────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │         CacheStorage             │
          │  ┌───────────┐  ┌────────────┐  │
          │  │ VectorDB   │  │ MetadataDB │  │
          │  │(GPU/CPU    │  │(optional   │  │
          │  │ hybrid)    │  │ MongoDB/   │  │
          │  │            │──│ SQLite)    │  │
          │  └───────────┘  └────────────┘  │
          └─────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                     InferencePipeline (existing, unmodified)              │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────┐               │
│  │  Stage 1    │───>│   Stage 2    │───>│   Stage 3     │               │
│  │  Vision     │    │   LLM        │    │   FlowMatch   │               │
│  └─────────────┘    └──────────────┘    └───────────────┘               │
└──────────────────────────────────────────────────────────────────────────┘
          ^                                         |
          |          Hook / Interceptor              |
          └─────── CacheOrchestrator ───────────────┘
```

### 4.2 与推理管线的接入方式

Cache 系统通过 **Interceptor 模式** 接入，不修改 `PI0Pytorch` 的内部代码：

```python
class InferenceInterceptor:
    """Wraps the inference pipeline, injecting cache checks between stages."""

    def __init__(self, policy: Policy, orchestrator: CacheOrchestrator):
        self.policy = policy
        self.orchestrator = orchestrator

    def infer(self, observation: dict) -> dict:
        """Cache-aware inference. Replaces direct policy.infer() calls."""
        timer = TimingContext()

        # --- Stage 1: Token Preparation ---
        with timer.stage("stage1_vision"):
            stage1_output = self.policy.run_stage1(observation)

        # --- CP1: Cache Check ---
        with timer.stage("cp1_check"):
            cp1_result = self.orchestrator.check(
                checkpoint=CheckpointID.CP1,
                context=stage1_output,
            )
        if cp1_result.hit:
            timer.record_event("cp1_hit")
            return cp1_result.cached_action

        # --- Stage 2: LLM Backbone ---
        with timer.stage("stage2_llm"):
            stage2_output = self.policy.run_stage2(stage1_output)

        # --- CP2: Cache Check ---
        with timer.stage("cp2_check"):
            cp2_result = self.orchestrator.check(
                checkpoint=CheckpointID.CP2,
                context=CacheContext(stage1=stage1_output, stage2=stage2_output),
            )
        if cp2_result.hit:
            timer.record_event("cp2_hit")
            if cp2_result.hit_type == HitType.FULL:
                return cp2_result.cached_action
            elif cp2_result.hit_type == HitType.WARM_START:
                # Partial flow matching from cached intermediate state
                with timer.stage("stage3_partial_flow"):
                    action = self.policy.run_stage3_from(
                        stage2_output,
                        start_x=cp2_result.cached_noisy_action,
                        start_t=cp2_result.cached_timestep,
                    )
                return action

        # --- Stage 3: Action Expert (full flow matching) ---
        with timer.stage("stage3_flow"):
            action, intermediates = self.policy.run_stage3(
                stage2_output, return_intermediates=True
            )

        # --- CP3: Predictive Cache Check ---
        with timer.stage("cp3_check"):
            cp3_result = self.orchestrator.check(
                checkpoint=CheckpointID.CP3,
                context=CacheContext(
                    stage1=stage1_output,
                    stage2=stage2_output,
                    action=action,
                ),
            )
        if cp3_result.hit:
            self.orchestrator.schedule_next_action(cp3_result.cached_next_action)

        # --- Write-back: populate cache (async, non-blocking) ---
        self.orchestrator.write_async(
            context=CacheContext(
                stage1=stage1_output,
                stage2=stage2_output,
                action=action,
                intermediates=intermediates,  # x_t at selected timesteps
            )
        )

        return action
```

关键设计点：
- `policy.run_stage1/2/3` 是对现有 `PI0Pytorch` 的薄封装，不改变其内部逻辑，只把 `sample_actions()` 的单体调用拆成三段可独立执行的子过程。
- `return_intermediates=True` 让 Stage 3 返回 flow matching 过程中选定时间步的 `x_t`，用于未来的 warm start cache。

---

## 5. 核心组件详细设计

### 5.1 CacheOrchestrator

总控组件。管理所有检查点的生命周期、协调 gate/search/judge 流程、处理异步写回。

```python
class CacheOrchestrator:
    def __init__(
        self,
        storage: CacheStorage,
        key_builder: QueryKeyBuilder,
        gate: GateFunction,
        judge: SimilarityJudge,
        config: CacheConfig,
        timer: SystemTimer,
    ):
        ...
        self._next_action_scheduled: Optional[CachedAction] = None
        self._write_queue: AsyncQueue = AsyncQueue()

    def should_skip_inference(self) -> Optional[dict]:
        """Called BEFORE inference starts. If CP3 from previous cycle
        scheduled a cached action, return it and skip entire inference."""
        if self._next_action_scheduled is not None:
            action = self._next_action_scheduled
            self._next_action_scheduled = None
            return action
        return None

    def check(self, checkpoint: CheckpointID, context: CacheContext) -> CacheResult:
        """Core cache check logic at a given checkpoint."""

        # Step 1: Gate - should we even search?
        with self.timer.measure(f"{checkpoint.name}_gate"):
            if not self.gate.should_search(checkpoint, context):
                return CacheResult.miss()

        # Step 2: Build query key
        with self.timer.measure(f"{checkpoint.name}_key_build"):
            query = self.key_builder.build(checkpoint, context)

        # Step 3: Search vector DB
        with self.timer.measure(f"{checkpoint.name}_search"):
            candidates = self.storage.search(query, top_k=self.config.top_k)

        # Step 4: Judge - is the best candidate good enough?
        with self.timer.measure(f"{checkpoint.name}_judge"):
            result = self.judge.evaluate(checkpoint, context, candidates)

        return result

    def write_async(self, context: CacheContext):
        """Non-blocking cache write. Runs on background thread."""
        self._write_queue.put(context)

    def schedule_next_action(self, action: CachedAction):
        """CP3 schedules an action for the next cycle."""
        self._next_action_scheduled = action
```

### 5.2 CacheStorage

存储层。包含 vector DB 和可选的 metadata DB。

```python
class CacheStorage:
    def __init__(self, config: StorageConfig):
        self.vector_db = VectorStore(config.vector_db)
        self.metadata_db = MetadataStore(config.metadata_db)  # optional

    def search(self, query: torch.Tensor, top_k: int) -> list[CacheCandidate]:
        """Search vector DB, return top-k candidates with metadata."""
        ids, distances = self.vector_db.search(query, top_k)
        entries = self.vector_db.get_payloads(ids)
        return [
            CacheCandidate(id=id, distance=d, entry=e)
            for id, d, e in zip(ids, distances, entries)
        ]

    def insert(self, entry: CacheEntry):
        """Insert new entry. Vector DB stores embedding + action data.
        Metadata DB stores auxiliary info (timestamp, task, episode, etc.)."""
        vector_id = self.vector_db.insert(
            vector=entry.query_key,
            payload=entry.to_payload(),  # action, intermediates, checkpoint_id
        )
        if self.metadata_db is not None:
            self.metadata_db.insert(vector_id, entry.metadata)

    def evict(self, policy: EvictionPolicy):
        """Remove entries based on eviction policy (LRU, LFU, quality-based)."""
        ...
```

### 5.3 VectorStore（GPU/CPU 混合）

```python
class VectorStore:
    """Hybrid GPU/CPU vector store for cache lookup.

    Design rationale:
    - Hot data (frequently accessed, recent) lives on GPU for fast search.
    - Cold data lives on CPU, searched when GPU partition misses.
    - Transfers between CPU/GPU use pinned memory + CUDA streams
      to avoid blocking the main inference CUDA stream.
    """

    def __init__(self, config: VectorStoreConfig):
        self.dim = config.embedding_dim
        self.gpu_capacity = config.gpu_capacity      # max entries on GPU
        self.cpu_capacity = config.cpu_capacity      # max entries on CPU

        # GPU partition: flat index for small-to-medium cache, IVF for large
        self.gpu_vectors: torch.Tensor  # [gpu_capacity, dim] on cuda
        self.gpu_payloads: list         # associated data
        self.gpu_count: int = 0

        # CPU partition: FAISS or custom index
        self.cpu_index: faiss.Index     # CPU-resident index
        self.cpu_payloads: list

        # Async transfer infrastructure
        self._transfer_stream = torch.cuda.Stream()

    def search(self, query: torch.Tensor, top_k: int) -> tuple[list[int], list[float]]:
        """Search GPU first, then CPU if needed.

        GPU search uses torch matmul on the dedicated stream.
        CPU search uses FAISS on a thread pool.
        Both can run concurrently.
        """
        # GPU search (fast, non-blocking on separate stream)
        gpu_ids, gpu_dists = self._search_gpu(query, top_k)

        # If GPU results are confident enough, skip CPU
        if gpu_dists and gpu_dists[0] < self.config.gpu_confidence_threshold:
            return gpu_ids, gpu_dists

        # CPU search (concurrent with GPU search in full mode)
        cpu_ids, cpu_dists = self._search_cpu(query.cpu(), top_k)

        # Merge results
        return self._merge_results(gpu_ids, gpu_dists, cpu_ids, cpu_dists, top_k)

    def _search_gpu(self, query: torch.Tensor, top_k: int):
        """Cosine similarity search on GPU using the transfer stream."""
        if self.gpu_count == 0:
            return [], []
        with torch.cuda.stream(self._transfer_stream):
            # query: [1, dim], gpu_vectors: [N, dim]
            sims = torch.mm(query, self.gpu_vectors[:self.gpu_count].T)
            topk = torch.topk(sims[0], min(top_k, self.gpu_count))
        self._transfer_stream.synchronize()
        return topk.indices.tolist(), topk.values.tolist()

    def promote_to_gpu(self, cpu_ids: list[int]):
        """Move frequently accessed CPU entries to GPU (async)."""
        with torch.cuda.stream(self._transfer_stream):
            # Pinned memory -> GPU transfer
            ...

    def demote_to_cpu(self, gpu_ids: list[int]):
        """Move cold GPU entries to CPU to free GPU memory."""
        ...
```

**GPU 显存预算管理**：VectorStore 通过 `gpu_capacity` 硬上限控制显存占用，实际占用 = `gpu_capacity * dim * sizeof(float16)` 字节。例如 10k entries x 1024 dim x 2 bytes = **20MB**，对 inference 的显存影响可忽略。

### 5.4 QueryKeyBuilder（可插拔）

```python
class QueryKeyBuilder(Protocol):
    """Converts stage outputs into a fixed-dimensional query vector."""

    def build(self, checkpoint: CheckpointID, context: CacheContext) -> torch.Tensor:
        """Returns a normalized query vector [1, embedding_dim]."""
        ...


class MeanPoolKeyBuilder(QueryKeyBuilder):
    """Baseline: mean-pool available embeddings, project to fixed dim."""

    def __init__(self, output_dim: int = 1024):
        self.projections = nn.ModuleDict({
            "vision": nn.Linear(..., output_dim),
            "prompt": nn.Linear(..., output_dim),
            "state": nn.Linear(..., output_dim),
            "command": nn.Linear(..., output_dim),
            "action": nn.Linear(..., output_dim),
        })

    def build(self, checkpoint, context):
        parts = []
        if context.stage1:
            parts.append(self.projections["vision"](context.stage1.vision_emb.mean(dim=1)))
            parts.append(self.projections["state"](context.stage1.state_emb))
        if context.stage2:
            parts.append(self.projections["command"](context.stage2.command_emb.mean(dim=1)))
        if context.action is not None:
            parts.append(self.projections["action"](context.action.mean(dim=1)))

        combined = torch.stack(parts).mean(dim=0)  # [1, output_dim]
        return F.normalize(combined, dim=-1)


class PlaceholderKeyBuilder(QueryKeyBuilder):
    """For early development: use raw state vector as key."""

    def build(self, checkpoint, context):
        return F.normalize(context.stage1.raw_state.float(), dim=-1)
```

设计为 Protocol，后续可替换为学习型 encoder 或其他方案，不影响系统其他部分。

### 5.5 GateFunction（可插拔）

决定是否在某个 checkpoint 启动检索。避免每次都搜索的开销。

```python
class GateFunction(Protocol):
    def should_search(self, checkpoint: CheckpointID, context: CacheContext) -> bool:
        ...

class AlwaysSearchGate(GateFunction):
    """Baseline: always search. For benchmarking overhead."""
    def should_search(self, checkpoint, context):
        return True

class IntervalGate(GateFunction):
    """Only search every N inference cycles."""
    def __init__(self, interval: int = 3):
        self.interval = interval
        self.counter = 0

    def should_search(self, checkpoint, context):
        self.counter += 1
        return self.counter % self.interval == 0

class StateChangeGate(GateFunction):
    """Search only when state change exceeds threshold.
    If robot barely moved since last check, cache result likely same -> skip search."""
    def __init__(self, threshold: float = 0.01):
        self.threshold = threshold
        self.last_state: Optional[torch.Tensor] = None

    def should_search(self, checkpoint, context):
        current = context.stage1.raw_state
        if self.last_state is None:
            self.last_state = current
            return True
        delta = (current - self.last_state).norm().item()
        if delta > self.threshold:
            self.last_state = current
            return True
        return False
```

### 5.6 SimilarityJudge（可插拔）

判定检索结果是否构成有效 hit。

```python
class SimilarityJudge(Protocol):
    def evaluate(
        self, checkpoint: CheckpointID, context: CacheContext, candidates: list[CacheCandidate]
    ) -> CacheResult:
        ...

class ThresholdJudge(SimilarityJudge):
    """Simple threshold-based judge with per-checkpoint thresholds."""

    def __init__(self, thresholds: dict[CheckpointID, float]):
        self.thresholds = thresholds
        # CP1 threshold should be stricter (higher similarity required)
        # because skipping more computation carries more risk.
        # Default: CP1=0.98, CP2=0.95, CP3=0.90

    def evaluate(self, checkpoint, context, candidates):
        if not candidates:
            return CacheResult.miss()

        best = candidates[0]
        threshold = self.thresholds[checkpoint]

        if best.distance >= threshold:  # cosine similarity
            return self._make_hit(checkpoint, best)
        return CacheResult.miss()

    def _make_hit(self, checkpoint, candidate):
        entry = candidate.entry
        if checkpoint == CheckpointID.CP2 and entry.has_intermediate:
            # Decide full hit vs warm start based on similarity
            if candidate.distance >= self.thresholds[CheckpointID.CP2_FULL]:
                return CacheResult.full_hit(entry.action)
            else:
                return CacheResult.warm_start(
                    cached_noisy_action=entry.intermediate_x_t,
                    cached_timestep=entry.intermediate_t,
                )
        return CacheResult.full_hit(entry.action)
```

### 5.7 CacheEntry 数据结构

```python
@dataclass
class CacheEntry:
    """What gets stored in the cache."""

    # Query key for retrieval
    query_key: torch.Tensor          # [embedding_dim], normalized

    # Checkpoint level this entry was created from
    checkpoint_id: CheckpointID

    # Core cached data
    action_chunk: torch.Tensor       # [action_horizon, action_dim], clean x_0

    # Optional: flow matching intermediates for warm start
    # Stores x_t at selected timesteps (e.g., t=0.7, 0.5, 0.3)
    intermediates: Optional[dict[float, torch.Tensor]] = None
    # intermediates = {0.7: x_0.7, 0.5: x_0.5, 0.3: x_0.3}

    # Context for CP3 predictive cache
    next_action_chunk: Optional[torch.Tensor] = None  # action from the NEXT cycle

    # Metadata
    timestamp: float = 0.0
    task_prompt: str = ""
    hit_count: int = 0               # for LFU eviction
    quality_score: float = 1.0       # for quality-based eviction
```

**Warm start 的中间状态选择**：不缓存所有 10 步的中间状态，只缓存 2-3 个关键时间点（如 t=0.7, 0.5, 0.3）。命中时选择离当前推理起点最近的缓存时间点。例如 CP2 warm start 时，如果缓存了 `x_0.3`，则从 t=0.3 开始执行 3 步 flow matching（而非全部 10 步），节省 70% 的 Stage 3 计算。

---

## 6. 数据流与时序

### 6.1 完整推理周期（无 cache hit）

```
Time ──────────────────────────────────────────────────────────────>

[GPU Main Stream]
│ Stage1 ││ Stage2 ││ Stage3 (10 denoise steps)          ││
│ Vision ││ LLM    ││ step1 step2 ... step10             ││
│        ││        ││                                     ││

[GPU Transfer Stream] (non-blocking)
         ││  CP1   ││       CP2        ││            CP3  ││  write-back
         ││ search ││      search      ││           check ││  (async)

[CPU Thread Pool]
         ││ CP1 CPU││   CP2 CPU search ││ CP3 CPU search  ││ metadata write
         ││ search ││  (if GPU miss)   ││                 ││
```

### 6.2 CP1 Hit 的时序

```
Time ──────────────────────────>

[GPU Main Stream]
│ Stage1 ││ (idle - stages 2,3 skipped)
│ Vision ││

[GPU Transfer Stream]
         ││ CP1 search ──> HIT!
         ││ load cached action from GPU memory

Total: Stage1 + CP1 latency only
```

### 6.3 CP2 Warm Start 的时序

```
Time ──────────────────────────────────────────────>

[GPU Main Stream]
│ Stage1 ││ Stage2 ││ Partial Stage3 (3 steps)    ││
│ Vision ││ LLM    ││ from cached x_0.3           ││

[GPU Transfer Stream]
         ││ CP1    ││ CP2 search ──> WARM START HIT
         ││ miss   ││ load cached x_0.3

Total: Stage1 + Stage2 + CP2 latency + 3 denoise steps (instead of 10)
```

### 6.4 CP3 Predictive Hit 的时序

```
Cycle N:                                             Cycle N+1:
│ Full inference │ CP3: match found ──> schedule │    │ Skip inference, use cached action │
                                                      │ (only run Stage1 for state update) │
```

---

## 7. 硬件资源分配策略

### 7.1 GPU 显存布局

```
GPU VRAM (e.g., 24GB)
├── Model weights (fixed)          ~5 GB  (PaliGemma 2B + Action Expert 300M, bf16)
├── KV Cache (per inference)       ~1 GB  (varies with sequence length)
├── Activations (transient)        ~2 GB  (peak during forward pass)
├── VectorStore GPU partition      ~20 MB (10k entries x 1024 dim x fp16)
├── Transfer buffers (pinned)      ~10 MB
└── Free                           ~16 GB
```

VectorStore 的 GPU 占用极小，不会成为瓶颈。

### 7.2 CUDA Stream 隔离

```python
class CacheHardwareManager:
    """Manages CUDA resources for cache operations."""

    def __init__(self):
        # Separate stream for cache operations, does NOT block inference
        self.cache_stream = torch.cuda.Stream(priority=-1)  # low priority
        # Pinned memory pool for CPU<->GPU transfers
        self.pinned_pool = PinnedMemoryPool(size_mb=32)

    @contextmanager
    def cache_context(self):
        """Execute cache operations on dedicated stream."""
        with torch.cuda.stream(self.cache_stream):
            yield

    def async_to_gpu(self, tensor_cpu: torch.Tensor) -> torch.Tensor:
        """Non-blocking CPU->GPU transfer via pinned memory."""
        pinned = self.pinned_pool.allocate(tensor_cpu.shape, tensor_cpu.dtype)
        pinned.copy_(tensor_cpu)
        gpu_tensor = torch.empty_like(pinned, device="cuda")
        with torch.cuda.stream(self.cache_stream):
            gpu_tensor.copy_(pinned, non_blocking=True)
        return gpu_tensor
```

### 7.3 CPU 线程分配

```
Thread 0 (main):       Inference orchestration
Thread 1:              CPU-side vector search (FAISS)
Thread 2:              Cache write-back (vector DB insert + metadata)
Thread 3 (optional):   Cache maintenance (eviction, compaction)
```

---

## 8. Cache 管理与动态优化

### 8.1 写入策略

- **在线写入**：每次正常推理完成后，异步写入 cache。不阻塞下一次推理。
- **离线预填充**：从训练数据或离线 rollout 中批量导入。
- **选择性写入**：不是所有推理结果都值得缓存。通过 `WriteFilter` 判断：
  - 如果当前状态与已有 cache 条目过于相似（< 某阈值），不写入（避免冗余）。
  - 如果动作置信度低（flow matching 收敛不好），不写入。

### 8.2 淘汰策略

```python
class EvictionPolicy(Protocol):
    def select_evictions(self, store: VectorStore, count: int) -> list[int]:
        ...

class CompositeEviction(EvictionPolicy):
    """Combine multiple signals for eviction."""

    def select_evictions(self, store, count):
        scores = []
        for entry in store.entries():
            score = (
                0.4 * recency_score(entry.timestamp)    # LRU component
                + 0.3 * frequency_score(entry.hit_count) # LFU component
                + 0.3 * entry.quality_score               # quality component
            )
            scores.append((entry.id, score))
        scores.sort(key=lambda x: x[1])
        return [id for id, _ in scores[:count]]
```

### 8.3 GPU/CPU 数据迁移策略

- **Promotion (CPU -> GPU)**：当某条 CPU 条目被命中超过 N 次，提升到 GPU partition。
- **Demotion (GPU -> CPU)**：当 GPU partition 满且有新的高频条目需要入驻时，将最低频的 GPU 条目降级到 CPU。
- **迁移在 `cache_stream` 上异步进行，不阻塞推理。**

---

## 9. 计时系统

```python
class SystemTimer:
    """Precise timing for all cache operations.

    Uses CUDA events for GPU operations, time.perf_counter_ns for CPU.
    All timings stored in a ring buffer, exportable for analysis.
    """

    def __init__(self, buffer_size: int = 10000):
        self._records: deque[TimingRecord] = deque(maxlen=buffer_size)

    @contextmanager
    def measure(self, name: str):
        """Time a block. Auto-detects GPU/CPU context."""
        if torch.cuda.is_available() and torch.cuda.current_stream() != torch.cuda.default_stream():
            # GPU timing with CUDA events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            yield
            end_event.record()
            end_event.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            # CPU timing
            start = time.perf_counter_ns()
            yield
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6

        self._records.append(TimingRecord(name=name, elapsed_ms=elapsed_ms, timestamp=time.time()))

    def summary(self, last_n: int = 100) -> dict[str, TimingStats]:
        """Aggregate timing stats per component."""
        ...

    def export_csv(self, path: str):
        """Export all timing records for external analysis."""
        ...
```

每个组件自动记录的计时项：

| Component | Timing Key | Description |
|-----------|-----------|-------------|
| Stage 1 | `stage1_vision` | Vision encoder + tokenization |
| Stage 2 | `stage2_llm` | LLM backbone autoregressive decode |
| Stage 3 | `stage3_flow` | Full flow matching (10 steps) |
| Stage 3 partial | `stage3_partial_flow` | Warm start flow matching |
| CP1 | `cp1_gate`, `cp1_key_build`, `cp1_search`, `cp1_judge` | Each sub-step |
| CP2 | `cp2_gate`, `cp2_key_build`, `cp2_search`, `cp2_judge` | Each sub-step |
| CP3 | `cp3_gate`, `cp3_key_build`, `cp3_search`, `cp3_judge` | Each sub-step |
| Write | `write_vectordb`, `write_metadata` | Async write-back |
| Transfer | `gpu_to_cpu`, `cpu_to_gpu` | Data migration |

---

## 10. 配置系统

```python
@dataclass
class CacheConfig:
    """Top-level cache system configuration."""

    enabled: bool = True

    # Per-checkpoint enable/disable
    cp1_enabled: bool = True
    cp2_enabled: bool = True
    cp3_enabled: bool = True

    # Retrieval
    top_k: int = 5                         # candidates per search
    embedding_dim: int = 1024              # query key dimension

    # Similarity thresholds (cosine similarity, higher = stricter)
    cp1_threshold: float = 0.98            # CP1 strictest: skipping most
    cp2_full_threshold: float = 0.96       # CP2 full hit
    cp2_warm_threshold: float = 0.90       # CP2 warm start (more lenient)
    cp3_threshold: float = 0.92            # CP3 predictive

    # Flow matching warm start
    intermediate_timesteps: list[float] = field(
        default_factory=lambda: [0.7, 0.5, 0.3]
    )  # which timesteps to cache x_t for

    # Storage
    vector_db: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    metadata_db: Optional[MetadataStoreConfig] = None

    # Hardware
    gpu_capacity: int = 10000              # max entries on GPU
    cpu_capacity: int = 100000             # max entries on CPU
    pinned_memory_mb: int = 32             # pinned memory pool

    # Write policy
    write_similarity_threshold: float = 0.99  # don't write if too similar to existing
    write_async: bool = True

    # Gate
    gate_type: str = "always"              # "always", "interval", "state_change"
    gate_interval: int = 1
    gate_state_threshold: float = 0.01

    # Timing
    timing_enabled: bool = True
    timing_buffer_size: int = 10000
```

---

## 11. 对现有代码的改动边界

### 必须修改的部分（最小侵入）

| 文件 | 改动 | 原因 |
|------|------|------|
| `models_pytorch/pi0_pytorch.py` | 将 `sample_actions()` 拆分出 `run_stage1()`, `run_stage2()`, `run_stage3()` 公共方法 | Cache 需要在 stage 之间插入检查点 |
| `models_pytorch/pi0_pytorch.py` | `run_stage3()` 增加 `return_intermediates` 参数和 `run_stage3_from(start_x, start_t)` 方法 | 支持 warm start 和中间状态缓存 |

### 不修改的部分

| 文件 | 说明 |
|------|------|
| `policies/policy.py` | Policy 类不变，InferenceInterceptor 在外层包装 |
| `models/pi0.py` | JAX 路径已关闭，不动 |
| `serving/websocket_policy_server.py` | 服务层不变，透明适配 |
| `training/` | 训练代码完全不动 |
| `transforms.py` | 数据变换不变 |

---

## 12. 文件结构规划

```
src/openpi/cache/                           # 新增的 cache 模块
├── __init__.py
├── config.py                               # CacheConfig, VectorStoreConfig, etc.
├── orchestrator.py                         # CacheOrchestrator 主控
├── interceptor.py                          # InferenceInterceptor (wraps Policy)
├── storage/
│   ├── __init__.py
│   ├── vector_store.py                     # VectorStore (GPU/CPU hybrid)
│   ├── metadata_store.py                   # MetadataStore (optional MongoDB/SQLite)
│   └── cache_entry.py                      # CacheEntry, CacheResult, etc.
├── components/
│   ├── __init__.py
│   ├── key_builder.py                      # QueryKeyBuilder protocol + implementations
│   ├── gate.py                             # GateFunction protocol + implementations
│   └── judge.py                            # SimilarityJudge protocol + implementations
├── hardware/
│   ├── __init__.py
│   ├── cuda_manager.py                     # CacheHardwareManager, stream management
│   └── memory_pool.py                      # PinnedMemoryPool
├── timing.py                               # SystemTimer, TimingRecord
├── maintenance/
│   ├── __init__.py
│   ├── eviction.py                         # EvictionPolicy implementations
│   ├── promotion.py                        # GPU/CPU data migration
│   └── writer.py                           # AsyncWriteWorker
└── tests/
    ├── test_orchestrator.py
    ├── test_vector_store.py
    ├── test_interceptor.py
    └── test_timing.py
```

---

## 13. 开发路线图

> 核心原则：**先搭骨架跑通，再在可运行的系统上做实验和优化。**
> 不要在没有端到端 pipeline 之前研究 similarity metric；不要在没有计时数据之前做性能优化。

---

### Step 0: 认识现有推理管线（前置，不写代码）

**目标**：对 PyTorch 推理路径建立精确的认知，所有后续工作基于此。

**工作内容**：

0.1. 精读 `src/openpi/models_pytorch/pi0_pytorch.py` 的 `sample_actions()` 方法，标注三个 stage 的代码边界（哪一行到哪一行是 vision/LLM/flow matching）。记录：
  - 每个 stage 的输入张量 shape、dtype、device
  - stage 之间传递的中间变量（KV cache 的具体结构、prefix tokens 的 shape）
  - flow matching 循环内部 `denoise_step()` 的输入输出

0.2. 精读 `src/openpi/policies/policy.py` 的 `infer()` 方法，理解 transforms 在 inference 时的实际调用顺序，确认 observation dict 在进入模型前的完整 key 列表和 shape。

0.3. 跑一次完整推理（用 `debug_pi05` config 或已有 checkpoint），用 `torch.cuda.Event` 手动测量三个 stage 各自的耗时。记录 baseline 数字：
  - Stage 1 (vision): __ ms
  - Stage 2 (LLM): __ ms
  - Stage 3 (flow matching, 10 steps total): __ ms，每步: __ ms
  - 端到端: __ ms

**产出**：一份简短的 baseline 报告（可以是 markdown 文件或 notebook），包含上述数字和代码边界标注。后续所有优化都以此为对照基线。

**这一步不能跳过的原因**：如果不清楚 stage 之间传递了什么张量、什么 shape，后续拆分 `sample_actions()` 时会反复踩坑。如果没有 baseline 延迟数据，后面无法判断 cache 引入的开销是否值得。

---

### Step 1: 拆分推理管线（最小侵入改动）

**目标**：将 `PI0Pytorch.sample_actions()` 拆成三段可独立调用的方法，同时保证原始调用路径不受影响。

**工作内容**：

1.1. 在 `pi0_pytorch.py` 中新增三个方法：

```python
def run_stage1(self, observation) -> Stage1Output:
    """Vision encoding + prefix KV cache construction.
    Extracts the code from sample_actions() up to and including
    the prefix forward pass that produces past_key_values."""
    ...

def run_stage2(self, stage1: Stage1Output) -> Stage2Output:
    """LLM backbone: generate low-level command tokens.
    For pi05=True, this is the autoregressive subtask prediction.
    For pi05=False (pi0), this stage may be trivial (no text generation)."""
    ...

def run_stage3(self, stage2: Stage2Output, *, return_intermediates=False) -> Stage3Output:
    """Action expert: full flow matching (10 Euler steps).
    If return_intermediates=True, also return x_t at selected timesteps."""
    ...
```

1.2. `sample_actions()` 改为内部顺序调用 `run_stage1 -> run_stage2 -> run_stage3`，逻辑完全等价，不改变任何计算。

1.3. 定义 stage 之间的数据结构：

```python
@dataclass
class Stage1Output:
    prefix_tokens: torch.Tensor
    prefix_masks: torch.Tensor
    past_key_values: tuple         # KV cache
    raw_state: torch.Tensor        # 原始 state vector，cache key 用
    # ... 其他 prefix 相关中间量

@dataclass
class Stage2Output:
    stage1: Stage1Output           # 包含 stage1 的输出
    command_tokens: torch.Tensor   # 生成的 subtask text tokens
    command_embedding: torch.Tensor # LLM 最后一层的 hidden state
    # ...

@dataclass
class Stage3Output:
    action_chunk: torch.Tensor     # [B, action_horizon, action_dim]
    intermediates: Optional[dict[float, torch.Tensor]] = None  # warm start 用
```

1.4. **验证**：跑同样的输入，对比拆分前后的输出张量，确认 `torch.allclose(original_output, split_output, atol=1e-5)`。这是硬性门控——不通过不进入下一步。

**关键注意事项**：
- Pi0.5 的 Stage 2（LLM subtask prediction）和 Pi0 的 Stage 2 逻辑不同。Pi0 没有 autoregressive text generation，LLM 只做一次前向。先只处理 Pi0.5 路径（`pi05=True`），Pi0 路径后续按需补充。
- `past_key_values` 的结构要原封不动传递，不要做任何 clone 或 reshape——这是最容易出数值 bug 的地方。

**产出**：修改后的 `pi0_pytorch.py`，通过数值一致性测试。

---

### Step 2: 计时系统

**目标**：实现 `SystemTimer`，为所有后续的性能量化提供基础设施。

**工作内容**：

2.1. 实现 `src/openpi/cache/timing.py`：
  - `SystemTimer` 类，支持 context manager `with timer.measure("name"):`
  - GPU 计时用 `torch.cuda.Event`，CPU 计时用 `time.perf_counter_ns`
  - Ring buffer 存储，`summary()` 输出均值/p50/p95/p99
  - `export_csv()` 导出原始记录

2.2. 将 timer 嵌入 Step 1 拆分后的三个 stage，替换 Step 0 的手动计时，验证数字一致。

**这一步必须在 cache 逻辑之前完成的原因**：后续每一步的开发都需要量化延迟。没有 timer，无法回答 "cache 检索花了多久" "是否值得" 这些问题。

**产出**：`timing.py` + 集成到 stage 调用中的计时。

---

### Step 3: Cache 数据结构与存储层

**目标**：实现 cache 的 "仓库" 部分——能存、能查、能取。此阶段不涉及与推理管线的集成。

**工作内容**：

3.1. 实现 `src/openpi/cache/storage/cache_entry.py`：
  - `CacheEntry` dataclass（query_key, action_chunk, checkpoint_id, metadata）
  - `CacheResult` dataclass（hit/miss, hit_type, cached_action, cached_noisy_action, cached_timestep）
  - `CheckpointID` enum（CP1, CP2, CP3）
  - `HitType` enum（FULL, WARM_START, PREDICTIVE）

3.2. 实现 `src/openpi/cache/storage/vector_store.py`：
  - **此阶段仅实现 CPU 版本**，使用 FAISS `IndexFlatIP`（内积，配合 L2 归一化等价于 cosine similarity）
  - `insert(vector, payload)` → id
  - `search(query, top_k)` → list of (id, similarity, payload)
  - `delete(id)`
  - `size()`, `clear()`
  - 单元测试：插入 1000 条随机向量，验证 search 返回正确的 top-k

3.3. 实现 `src/openpi/cache/config.py`：
  - `CacheConfig` dataclass，包含所有可调参数
  - 提供 `default_config()` 和 `debug_config()` 工厂方法

**这一步与推理无关**：存储层是纯粹的数据结构，可以独立开发和测试。不需要加载模型，不需要 GPU。

**产出**：数据结构 + CPU VectorStore + 单元测试通过。

---

### Step 4: Orchestrator 骨架 + Interceptor

**目标**：将 cache 检查逻辑与推理管线连接起来，实现端到端的 cache 工作流。此阶段使用最简单的组件实现（PlaceholderKeyBuilder + AlwaysSearchGate + ThresholdJudge），**只开启 CP2**。

**为什么先做 CP2 而非 CP1**：
- CP2 在 LLM 之后，此时已有 subtask 信息，cache 的语义最清晰（"相同场景+相同指令→相同动作"）。
- CP2 命中跳过的是 flow matching（最纯粹的数值计算），不涉及跳过语义理解，风险最低。
- CP1 跳过 subtask 预测，风险高，不适合作为第一个验证点。
- CP3 是预测性的，逻辑更复杂，不适合首轮。

**工作内容**：

4.1. 实现 `src/openpi/cache/components/key_builder.py`：
  - `QueryKeyBuilder` Protocol
  - `PlaceholderKeyBuilder`：直接用 `stage2_output.command_embedding` 做 mean pool + L2 normalize，或者更简单地用 `raw_state` concatenate `command_embedding` 然后 normalize。具体维度取决于 Step 0 中记录的张量 shape。

4.2. 实现 `src/openpi/cache/components/gate.py`：
  - `GateFunction` Protocol
  - `AlwaysSearchGate`：永远返回 True

4.3. 实现 `src/openpi/cache/components/judge.py`：
  - `SimilarityJudge` Protocol
  - `ThresholdJudge`：cosine similarity > threshold → hit

4.4. 实现 `src/openpi/cache/orchestrator.py`：
  - `CacheOrchestrator`：组合 key_builder + gate + judge + storage
  - `check()` 方法：gate → build key → search → judge
  - `write_async()` 方法：先用同步写入（async 在 Step 7 优化）

4.5. 实现 `src/openpi/cache/interceptor.py`：
  - `InferenceInterceptor`：包装 Policy，在 stage 之间注入 cache check
  - 此阶段只在 Stage 2 之后注入 CP2 check

4.6. **端到端测试**：
  - 加载模型，跑 10 次相同输入 → 第 1 次 miss，第 2-10 次应该 hit（输入完全相同）
  - 跑 10 次不同输入 → 全部 miss
  - 验证 hit 时返回的 action 与正常推理结果的 L2 距离（应该 = 0，因为是完全相同的输入）

**产出**：可运行的端到端 cache 系统（仅 CP2），通过上述测试。

---

### Step 5: 基础实验——Cache 可行性验证

**目标**：回答核心问题——"对于相似但不完全相同的输入，cache hit 返回的 action 质量如何？" 这决定了整个 cache 系统是否有意义。

> **这是整个项目的第一个关键实验节点。** 如果实验结果表明相似输入的 action 差异过大，整个 cache 思路需要重新评估。不要在这个实验之前投入更多开发工作。

**实验设计**：

5.1. **数据准备**：收集一组推理 episode（100-500 步），记录每一步的：
  - 输入 observation（images, state, prompt）
  - Stage 1 输出（vision embedding）
  - Stage 2 输出（command embedding）
  - 最终 action chunk
  - 将所有上述数据保存到磁盘（HDF5 或 pickle）

5.2. **实验 A：State 空间中的 action 连续性**
  - 对记录的 episode，计算所有 step 两两之间的 state cosine similarity
  - 计算对应的 action L2 距离
  - 画 scatter plot：x=state_similarity, y=action_distance
  - **期望看到**：state similarity 高时 action distance 低（正相关）
  - **如果看不到这个趋势**：cache 思路存在根本问题

5.3. **实验 B：Cache hit 的 action 质量**
  - 用 Step 4 的系统，逐步降低 CP2 threshold（从 0.99 到 0.80）
  - 记录每个 threshold 下的：hit rate, action L2 error (vs 正常推理), 延迟节省
  - 画三条曲线：threshold vs hit_rate, threshold vs action_error, threshold vs latency_saving
  - **寻找 sweet spot**：action error 可接受（< 某个值）的前提下 hit rate 最大化

5.4. **实验 C：不同 query key 的区分度**
  - 对比几种 key 构建方式的检索质量：
    - (a) raw state vector only
    - (b) command embedding only
    - (c) state + command concatenation
    - (d) vision embedding mean pool
  - 指标：precision@k（top-k 检索结果中，action 真正相近的比例）
  - **这个实验指导后续 QueryKeyBuilder 的设计**

**产出**：实验报告，包含上述图表和结论。决定是否继续，以及初步确定 threshold 范围和 key builder 方向。

---

### Step 6: CP1 和 CP3 实现

**前置条件**：Step 5 实验结果正面（cache 可行性得到验证）。

**工作内容**：

6.1. **CP1 实现**：
  - 在 Stage 1 之后注入 cache check
  - Key builder 需要处理 vision embedding（Step 5 实验 C 会给出方向）
  - CP1 使用更严格的 threshold（默认 0.98）
  - 测试：相同场景相同 prompt 应该 hit，更换物体或 prompt 应该 miss

6.2. **CP3 实现**：
  - 在 Stage 3 之后注入检查
  - `schedule_next_action()` 机制：在 orchestrator 中维护一个 `_next_action_scheduled` 槽位
  - `should_skip_inference()`：在每个 cycle 开始前检查是否有预调度的 action
  - CP3 的 key 需要包含 action chunk 信息（因为是预测"下一步"）
  - 需要维护 **连续 action 序列的对应关系**——entry 中增加 `next_entry_id` 字段，指向时间上紧接的下一个 entry

6.3. **CP3 特殊问题**：CP3 的 cache entry 需要记录 "当前 action → 下一步 action" 的对应关系。这意味着：
  - 写入 cache 时，需要等到**下一个** cycle 的 action 产出后，才能补全当前 entry 的 `next_action_chunk` 字段
  - 实现一个 `DeferredWriter`：在 cycle N 写入 entry（不含 next），在 cycle N+1 回填 next_action_chunk

6.4. **实验**：
  - 在 episode 上统计 CP1/CP2/CP3 各自的 hit rate
  - 量化三个检查点的命中时的延迟节省
  - CP3 的 predictive accuracy：预调度的 action 与实际推理出的 action 的 L2 距离

**产出**：三检查点完整系统 + 各检查点的命中率和延迟报告。

---

### Step 7: Flow Matching Warm Start

**前置条件**：Step 6 完成，CP2 full hit 路径已验证。

**工作内容**：

7.1. 修改 `run_stage3()`：
  - `return_intermediates=True` 时，在 flow matching 循环中保存选定时间步的 `x_t`
  - 默认保存 t=0.7, 0.5, 0.3 三个点（可配置）
  - 保存的张量 shape = `[B, action_horizon, action_dim]`，与 noise 相同

7.2. 新增 `run_stage3_from(stage2_output, start_x, start_t)`：
  - 从 `start_x` 和 `start_t` 开始执行剩余的 Euler steps
  - 例如 `start_t=0.3` 时只跑 3 步（0.3 → 0.2 → 0.1 → 0.0），而非 10 步

7.3. CP2 judge 增加 warm start 判定逻辑：
  - similarity > `cp2_full_threshold` → FULL hit
  - `cp2_warm_threshold` < similarity < `cp2_full_threshold` → WARM_START hit
  - similarity < `cp2_warm_threshold` → miss

7.4. **关键实验：Warm Start 精度 vs 速度 tradeoff**

  这是第二个关键实验节点。

  - 对同一组输入，分别跑：
    - (a) 完整 10 步 flow matching（baseline）
    - (b) 从自身的 cached x_0.7 warm start（3 步跳过）
    - (c) 从自身的 cached x_0.5 warm start（5 步跳过）
    - (d) 从自身的 cached x_0.3 warm start（7 步跳过）
  - 测量 action L2 error vs baseline
  - 这个实验的 "自身 cached" 意味着用**完全相同输入**的中间状态，隔离 warm start 本身的误差（不涉及 state 相似度问题）

  - 然后用**相似但不同**输入的 cached x_t 做 warm start：
    - 从相似 state 的 episode 中取 cached x_0.5
    - 用当前 observation 的 velocity field 继续 denoise
    - 测量 action L2 error
  - **期望**：error 在可接受范围内，且比 "直接用 cache action 不做 flow matching" 更小

  - 画 trade-off 图：x=跳过的步数, y=action_error, 多条线表示不同 state similarity 级别

**产出**：warm start 实现 + trade-off 实验数据，确定默认的 warm start 时间点和 threshold。

---

### Step 8: 系统效率优化——异步与硬件

**前置条件**：Step 7 完成，功能正确性已验证。到这一步才做性能优化，因为优化之前需要精确的 timing 数据来指导投入方向。

**优化决策流程**：先用 Step 2 的 timer 生成完整的延迟分解报告，识别瓶颈所在，然后有针对性地优化。不要凭猜测优化。

**可能的优化方向**（按预期收益排序）：

8.1. **异步 cache 写入**（几乎肯定需要）：
  - 当前 Step 4 的写入是同步的，会阻塞推理
  - 实现 `AsyncWriteWorker`：后台线程从队列中消费写入请求
  - 使用 `threading.Thread` + `queue.Queue`（不需要 multiprocessing，因为写入是 I/O bound）
  - 验证：写入延迟从推理关键路径中消失

8.2. **GPU VectorStore**（如果 CPU 搜索是瓶颈）：
  - 查看 timer 报告中 `cp*_search` 的延迟
  - 如果 CPU FAISS 搜索延迟 > 1ms 且 cache 条目 > 10k，考虑 GPU partition
  - 实现 `torch.mm` cosine similarity search on dedicated CUDA stream
  - 使用独立 stream 避免阻塞主推理 stream
  - 验证：search 延迟下降，主 stream 推理延迟不受影响

8.3. **CUDA Stream 隔离**（如果 cache 操作阻塞了推理）：
  - 实现 `CacheHardwareManager`
  - Cache 的所有 GPU 操作（search, key 构建中的 projection）在 `cache_stream` 上执行
  - Pinned memory pool 用于 CPU↔GPU 数据传输

8.4. **Gate 优化**（如果 cache check 本身成为瓶颈）：
  - 实现 `StateChangeGate`：state 变化小于阈值时跳过搜索
  - 实现 `IntervalGate`：每 N 次推理只搜索一次
  - 这可以大幅减少搜索频率，在 hit rate 低的场景下尤其有用

8.5. **淘汰策略**（如果 cache 增长导致搜索变慢）：
  - 实现 `CompositeEviction`（LRU + LFU + quality）
  - 设置 capacity 上限，定期淘汰
  - 淘汰在后台线程执行

**产出**：优化后的系统 + 优化前后的对比延迟报告。

---

### Step 9: Query Key 研究（实验密集型）

**前置条件**：Step 8 完成，系统性能在可接受范围。此时有一个稳定运行的 cache 系统，可以作为实验平台。

**为什么放在这里而非更早**：Query key 的研究需要在真实运行的系统上做，需要真实的 hit/miss 数据、真实的延迟数字。在系统跑通之前研究 key 是空中楼阁。

**实验方向**：

9.1. **Key 信息源消融实验**：
  - 在已有的 episode 数据上，对比不同信息组合作为 key 的检索质量：

  | Key 组合 | 维度 | Precision@5 | Recall@5 | 计算开销 |
  |---------|------|-------------|----------|---------|
  | raw_state | 32 | ? | ? | 极低 |
  | state + prompt_hash | 32+64 | ? | ? | 低 |
  | vision_emb (mean pool) | 2048 | ? | ? | 中 |
  | command_emb (mean pool) | 2048 | ? | ? | 中 |
  | state + command_emb | 32+2048 | ? | ? | 中 |
  | learned projection | 128/256/512 | ? | ? | 需训练 |

  - 其中 "Precision@5" 定义为：top-5 检索到的 entry 中，其 action 与当前推理 action 的 L2 距离 < epsilon 的比例

9.2. **Learned Key Builder**（如果简单方法效果不够）：
  - 训练一个小 projection head（2-3 层 MLP），输入为 stage output 的 concatenation，输出为低维 key
  - 训练目标：contrastive loss——相似 state 的 key 距离近，不同 state 的 key 距离远
  - 训练数据来自离线 episode 收集
  - 约束：projection head 的推理延迟 < 0.5ms（否则不如不用 cache）

9.3. **不同检查点的最优 key 可能不同**：
  - CP1 的 key 没有 command 信息，可能需要更多 vision 信息
  - CP2 的 key 有 command，state 信息的权重可能可以降低
  - CP3 的 key 需要 action 信息来预测后续
  - 每个 checkpoint 独立调参

**产出**：每个 checkpoint 的最优 key builder 方案 + 实验数据支撑。

---

### Step 10: 离线预填充管道

**前置条件**：Step 9 完成，key builder 方案确定。

**工作内容**：

10.1. 实现 `scripts/prefill_cache.py`：
  - 从训练数据或离线 rollout 中加载 episodes
  - 对每个 timestep 跑模型推理（或直接从保存的 episode 数据中读取）
  - 构建 key，写入 VectorStore
  - 支持增量预填充（检查已有 entry，避免重复）

10.2. 序列化/反序列化：
  - VectorStore 支持 `save(path)` 和 `load(path)`
  - 预填充后保存到磁盘，推理时直接加载，避免每次冷启动

10.3. **预填充质量验证**：
  - 用预填充的 cache 跑 inference episode
  - 对比有 cache vs 无 cache 的 action 轨迹差异
  - 量化预填充 cache 的 hit rate（期望远高于空 cache 在线积累）

**产出**：离线预填充脚本 + 序列化支持。

---

### Step 11: 集成测试与鲁棒性

**工作内容**：

11.1. **长时间运行稳定性测试**：
  - 运行 1000+ 步的 episode，监控：
    - 内存是否泄漏（cache 增长是否受控）
    - 延迟是否稳定（是否有逐渐变慢的趋势）
    - hit rate 是否合理变化

11.2. **异常场景测试**：
  - cache 为空时的行为（全 miss，正常推理）
  - cache 满时的淘汰行为
  - 输入异常（全黑图像、空 prompt）时不 crash
  - GPU OOM 时 graceful fallback 到纯 CPU

11.3. **A/B 对比框架**：
  - 实现 `--cache_enabled` flag，一键开关 cache
  - 对比相同 episode 下 cache on/off 的：
    - 端到端延迟（mean, p95）
    - Action 质量（与 no-cache baseline 的 L2 距离）
    - GPU 利用率

**产出**：稳定性测试报告 + A/B 对比数据。

---

### Step 12: 进阶功能（按需）

以下各项相互独立，按需求优先级选择实现：

12.1. **Metadata DB（MongoDB/SQLite）**：
  - 当需要存储 vector DB 之外的丰富信息时引入（task name, episode id, success/failure label 等）
  - 用于 cache 分析和离线质量评估
  - 不在关键路径上，不影响推理延迟

12.2. **GPU/CPU 动态迁移**：
  - Promotion/Demotion 策略
  - 基于访问频率的自动数据分层

12.3. **学习型 Gate Function**：
  - 训练一个二分类器预测 "当前 state 是否可能 cache hit"
  - 输入：state delta (vs 上次)、task prompt hash、cache 统计信息
  - 减少无意义的搜索开销

12.4. **分布式 Cache**：
  - 多机器人共享 vector DB
  - 需要考虑网络延迟和一致性

12.5. **Cache-aware 训练**：
  - 在训练时引入 cache-hit 模拟，让模型适应偶尔跳步的场景
  - 长期方向，需要大量实验

---

### 开发依赖关系总图

```
Step 0: 认识推理管线
  │
  ▼
Step 1: 拆分 sample_actions()  ──────────────────────────────────┐
  │                                                               │
  ▼                                                               │
Step 2: 计时系统                                                   │
  │                                                               │
  ├──────────────────┐                                            │
  ▼                  ▼                                            │
Step 3: 数据结构    (并行开发)                                      │
  │                                                               │
  ▼                                                               │
Step 4: Orchestrator + Interceptor (CP2 only) ◄───────────────────┘
  │
  ▼
Step 5: ★ 可行性实验 ★  ── 如果失败 ──> 重新评估整体方案
  │ (通过)
  ▼
Step 6: CP1 + CP3 实现
  │
  ▼
Step 7: Warm Start 实现 + ★ 精度实验 ★
  │
  ▼
Step 8: 系统效率优化 (async, GPU, stream)
  │
  ▼
Step 9: ★ Query Key 研究 ★ (实验密集)
  │
  ▼
Step 10: 离线预填充
  │
  ▼
Step 11: 集成测试
  │
  ▼
Step 12: 进阶功能 (按需)
```

**★ 标记的步骤是关键实验节点**，其结论直接决定后续工作的方向甚至是否继续。

---

### 各步骤预估工作量

| Step | 工作类型 | 主要产出 | 预估开发量 |
|------|---------|---------|-----------|
| 0 | 阅读+分析 | Baseline 报告 | 1-2 天 |
| 1 | 代码改造 | 拆分后的 pi0_pytorch.py | 2-3 天 |
| 2 | 基础设施开发 | timing.py | 1 天 |
| 3 | 基础设施开发 | 数据结构 + VectorStore | 2 天 |
| 4 | 核心开发 | Orchestrator + Interceptor | 3-4 天 |
| 5 | **实验** | 可行性报告 | 3-5 天 |
| 6 | 核心开发 | CP1 + CP3 | 3-4 天 |
| 7 | 开发+**实验** | Warm start + tradeoff 数据 | 4-5 天 |
| 8 | 性能优化 | 异步/GPU/Stream | 3-5 天 |
| 9 | **实验** | Key builder 研究 | 5-7 天 |
| 10 | 工具开发 | 预填充脚本 | 2-3 天 |
| 11 | 测试 | 稳定性+A/B 报告 | 2-3 天 |
| 12 | 进阶 | 按需 | 不定 |

> **注**：以上时间估算不含等待实验跑完和分析结果的时间。实际实验周期取决于 GPU 资源和 episode 数据量。Step 3 和 Step 2 之间无依赖，可以并行开发。

---

## 14. 关键设计决策与 Tradeoff 记录

| 决策 | 选择 | 替代方案 | 理由 |
|------|------|---------|------|
| 接入方式 | Interceptor 外包装 | 修改 PI0Pytorch 内部 | 解耦优先，便于回退和 A/B 测试 |
| GPU vector search | torch.mm 在独立 CUDA stream | FAISS GPU index | 更可控，避免 FAISS GPU 的 CUDA 上下文竞争 |
| 中间状态缓存 | 选择性时间点 (2-3个) | 全部 10 步 | 存储效率，10步全存 action_dim=32 也只多几 KB，但检索判定更复杂 |
| Query key | Protocol 接口，初期 raw state | 固定方案 | 目前信息不足以确定最优方案，保持灵活 |
| CP1 阈值最严 | 0.98 | 统一阈值 | CP1 跳过最多计算，错误代价最高 |
| 写入去重 | 相似度检查 | 全写 | 避免 cache 膨胀，保持检索效率 |
