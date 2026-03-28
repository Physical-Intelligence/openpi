# Pi0.5 Inference Cache System - Architecture Specification

> Version: 0.1 (Draft)
> Status: Design Phase
> Scope: PyTorch inference pipeline only (JAX path disabled)

---

## 1. System Goals

Introduce a multi-level cache system into the Pi0.5 inference pipeline to reduce redundant computation by reusing historical results, thereby lowering end-to-end latency. The design adheres to the following principles:

1. **Decoupled from inference pipeline**: The cache system operates as an external plugin, hooking into the inference pipeline via an interceptor pattern without modifying existing inference code internals.
2. **Multi-level progressive hit**: Cache checkpoints are placed at three key positions in the pipeline — earlier hits save more computation.
3. **Hardware-aware**: Vector DB data is intelligently distributed between GPU and CPU, with async transfers that do not block inference computation.
4. **Precise timing**: Each component (retrieval, judgment, data transfer) is independently timed to support informed performance optimization decisions.
5. **Incremental implementation**: Start with single-machine, single-task repetitive scenarios, then gradually extend to multi-task/multi-robot/distributed settings.

---

## 2. Three-Stage Inference Pipeline Model

Based on the Pi0.5 PyTorch inference path (`src/openpi/models_pytorch/pi0_pytorch.py`), inference is divided into three stages:

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

**Estimated compute per stage (single inference)**:

| Stage | Primary Computation | Parameters | Characteristics |
|-------|---------------------|------------|-----------------|
| 1. Token Prep | SigLIP forward + tokenize | ~400M | Single forward pass, parallelizable |
| 2. LLM Backbone | Gemma 2B autoregressive decode | ~2B | Autoregressive, sequential dependency |
| 3. Action Expert | 10x Gemma 300M forward | ~300M x10 | Iterative, partially skippable |

---

## 3. Checkpoint Semantics

### CP1: After Vision

- **Trigger**: Stage 1 complete; prefix tokens and KV cache generated.
- **Available information**: Vision embedding, prompt embedding, state embedding.
- **Hit behavior**: Skip Stage 2 + Stage 3, directly output cached action chunk.
- **Savings**: Maximum (skip LLM decoding + all flow matching).
- **Risk**: Highest — skips subtask prediction. If the scene has changed subtly (e.g., an object was removed), the cached subtask may no longer be correct.
- **Applicable scenario**: Highly repetitive operations (e.g., the same action on an assembly line).

### CP2: After LLM Backbone

- **Trigger**: Stage 2 complete; low-level command (subtask text tokens) generated.
- **Available information**: All CP1 information + low-level command embedding.
- **Hit behavior (two modes)**:
  - **Full hit**: Skip all of Stage 3, directly output cached action chunk.
  - **Partial hit (warm start)**: Use cached intermediate state `x_t` (t < 1.0) as flow matching starting point, skipping some denoising steps.
- **Savings**: Medium (skip all or part of flow matching).
- **Risk**: Medium — subtask was computed by the current inference; cached action has higher consistency with the current scene.
- **Applicable scenario**: Reuse of the same subtask in similar scenes.

### CP3: After Action Expert

- **Trigger**: Stage 3 complete; current cycle's action chunk generated.
- **Available information**: All information (vision + prompt + state + command + action chunk).
- **Hit behavior**: Does NOT affect the current cycle's output. Determines whether the **next inference cycle** can be skipped, directly executing cached subsequent action chunks.
- **Savings**: Maximum (skip an entire next inference).
- **Risk**: Medium — depends on the accuracy of future state prediction.
- **Applicable scenario**: Scenarios where consecutive action sequences exhibit temporal locality (e.g., the middle phase of a long object transport).

### Checkpoint Relationship Diagram

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

## 4. System Architecture

### 4.1 Top-Level Component Diagram

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

### 4.2 Integration with the Inference Pipeline

The cache system hooks in via the **Interceptor pattern**, without modifying `PI0Pytorch` internals:

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

Key design points:
- `policy.run_stage1/2/3` are thin wrappers around the existing `PI0Pytorch`, breaking the monolithic `sample_actions()` call into three independently executable sub-processes without changing internal logic.
- `return_intermediates=True` causes Stage 3 to return `x_t` at selected timesteps during flow matching, for future warm start caching.

---

## 5. Core Component Detailed Design

### 5.1 CacheOrchestrator

The master controller. Manages the lifecycle of all checkpoints, coordinates the gate/search/judge workflow, and handles async write-back.

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

The storage layer. Contains a vector DB and an optional metadata DB.

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

### 5.3 VectorStore (GPU/CPU Hybrid)

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

**GPU VRAM budget management**: VectorStore controls VRAM usage via a hard `gpu_capacity` cap. Actual usage = `gpu_capacity * dim * sizeof(float16)` bytes. For example, 10k entries x 1024 dim x 2 bytes = **20MB**, negligible impact on inference VRAM.

### 5.4 QueryKeyBuilder (Pluggable)

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

Designed as a Protocol so it can be swapped for a learned encoder or other approaches later without affecting the rest of the system.

### 5.5 GateFunction (Pluggable)

Decides whether to initiate a search at a given checkpoint. Avoids the overhead of searching every time.

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

### 5.6 SimilarityJudge (Pluggable)

Determines whether search results constitute a valid hit.

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

### 5.7 CacheEntry Data Structure

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

**Intermediate state selection for warm start**: Rather than caching all 10 intermediate states, only 2-3 key timesteps are cached (e.g., t=0.7, 0.5, 0.3). On hit, the cached timestep closest to the current inference starting point is selected. For example, with a CP2 warm start using cached `x_0.3`, flow matching executes only 3 steps (instead of 10), saving 70% of Stage 3 computation.

---

## 6. Data Flow and Timing

### 6.1 Full Inference Cycle (No Cache Hit)

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

### 6.2 CP1 Hit Timing

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

### 6.3 CP2 Warm Start Timing

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

### 6.4 CP3 Predictive Hit Timing

```
Cycle N:                                             Cycle N+1:
│ Full inference │ CP3: match found ──> schedule │    │ Skip inference, use cached action │
                                                      │ (only run Stage1 for state update) │
```

---

## 7. Hardware Resource Allocation Strategy

### 7.1 GPU VRAM Layout

```
GPU VRAM (e.g., 24GB)
├── Model weights (fixed)          ~5 GB  (PaliGemma 2B + Action Expert 300M, bf16)
├── KV Cache (per inference)       ~1 GB  (varies with sequence length)
├── Activations (transient)        ~2 GB  (peak during forward pass)
├── VectorStore GPU partition      ~20 MB (10k entries x 1024 dim x fp16)
├── Transfer buffers (pinned)      ~10 MB
└── Free                           ~16 GB
```

VectorStore GPU usage is minimal and will not become a bottleneck.

### 7.2 CUDA Stream Isolation

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

### 7.3 CPU Thread Allocation

```
Thread 0 (main):       Inference orchestration
Thread 1:              CPU-side vector search (FAISS)
Thread 2:              Cache write-back (vector DB insert + metadata)
Thread 3 (optional):   Cache maintenance (eviction, compaction)
```

---

## 8. Cache Management and Dynamic Optimization

### 8.1 Write Strategy

- **Online writes**: After each normal inference completes, async write to cache. Does not block the next inference.
- **Offline pre-fill**: Batch import from training data or offline rollouts.
- **Selective writes**: Not all inference results are worth caching. A `WriteFilter` decides:
  - If the current state is too similar to an existing cache entry (< threshold), don't write (avoid redundancy).
  - If action confidence is low (poor flow matching convergence), don't write.

### 8.2 Eviction Strategy

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

### 8.3 GPU/CPU Data Migration Strategy

- **Promotion (CPU -> GPU)**: When a CPU entry is hit more than N times, promote it to the GPU partition.
- **Demotion (GPU -> CPU)**: When the GPU partition is full and a new high-frequency entry needs to be added, demote the lowest-frequency GPU entry to CPU.
- **Migration runs asynchronously on `cache_stream`, not blocking inference.**

---

## 9. Timing System

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

Timing keys recorded automatically by each component:

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

## 10. Configuration System

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

## 11. Modification Boundary for Existing Code

### Required Modifications (Minimal Invasion)

| File | Change | Reason |
|------|--------|--------|
| `models_pytorch/pi0_pytorch.py` | Split `sample_actions()` into `run_stage1()`, `run_stage2()`, `run_stage3()` public methods | Cache needs to insert checkpoints between stages |
| `models_pytorch/pi0_pytorch.py` | Add `return_intermediates` parameter to `run_stage3()` and a `run_stage3_from(start_x, start_t)` method | Support warm start and intermediate state caching |

### Unmodified Parts

| File | Notes |
|------|-------|
| `policies/policy.py` | Policy class unchanged; InferenceInterceptor wraps it externally |
| `models/pi0.py` | JAX path disabled, untouched |
| `serving/websocket_policy_server.py` | Serving layer unchanged, transparent adaptation |
| `training/` | Training code completely untouched |
| `transforms.py` | Data transforms unchanged |

---

## 12. Planned File Structure

```
src/openpi/cache/                           # New cache module
├── __init__.py
├── config.py                               # CacheConfig, VectorStoreConfig, etc.
├── orchestrator.py                         # CacheOrchestrator master controller
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

## 13. Development Roadmap

> Core principle: **Build the skeleton and get it running first, then experiment and optimize on the working system.**
> Do not research similarity metrics before having an end-to-end pipeline; do not optimize performance before having timing data.

---

### Step 0: Understand the Existing Inference Pipeline (Prerequisite, No Code)

**Goal**: Build a precise understanding of the PyTorch inference path. All subsequent work depends on this.

**Tasks**:

0.1. Carefully read `src/openpi/models_pytorch/pi0_pytorch.py`'s `sample_actions()` method. Annotate the code boundaries of the three stages (which lines are vision/LLM/flow matching). Record:
  - Input tensor shapes, dtypes, and devices for each stage
  - Intermediate variables passed between stages (KV cache structure, prefix token shapes)
  - `denoise_step()` inputs and outputs within the flow matching loop

0.2. Read `src/openpi/policies/policy.py`'s `infer()` method. Understand the actual calling order of transforms during inference. Confirm the complete key list and shapes of the observation dict before it enters the model.

0.3. Run a complete inference (using the `debug_pi05` config or an existing checkpoint). Manually measure each stage's latency with `torch.cuda.Event`. Record baseline numbers:
  - Stage 1 (vision): __ ms
  - Stage 2 (LLM): __ ms
  - Stage 3 (flow matching, 10 steps total): __ ms, per step: __ ms
  - End-to-end: __ ms

**Deliverable**: A short baseline report (markdown file or notebook) containing the above numbers and code boundary annotations. All subsequent optimizations are measured against this baseline.

**Why this step cannot be skipped**: Without knowing what tensors pass between stages and their shapes, splitting `sample_actions()` later will be error-prone. Without baseline latency data, there's no way to judge whether cache overhead is worthwhile.

---

### Step 1: Split the Inference Pipeline (Minimal Invasion)

**Goal**: Split `PI0Pytorch.sample_actions()` into three independently callable methods while ensuring the original call path remains unaffected.

**Tasks**:

1.1. Add three new methods to `pi0_pytorch.py`:

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

1.2. Refactor `sample_actions()` to internally call `run_stage1 -> run_stage2 -> run_stage3` sequentially, with fully equivalent logic and no computation changes.

1.3. Define inter-stage data structures:

```python
@dataclass
class Stage1Output:
    prefix_tokens: torch.Tensor
    prefix_masks: torch.Tensor
    past_key_values: tuple         # KV cache
    raw_state: torch.Tensor        # Original state vector, for cache key
    # ... other prefix-related intermediates

@dataclass
class Stage2Output:
    stage1: Stage1Output           # Contains stage1 output
    command_tokens: torch.Tensor   # Generated subtask text tokens
    command_embedding: torch.Tensor # LLM last layer hidden state
    # ...

@dataclass
class Stage3Output:
    action_chunk: torch.Tensor     # [B, action_horizon, action_dim]
    intermediates: Optional[dict[float, torch.Tensor]] = None  # For warm start
```

1.4. **Verification**: Run the same input and compare output tensors before and after the split. Confirm `torch.allclose(original_output, split_output, atol=1e-5)`. This is a hard gate — do not proceed to the next step until this passes.

**Key considerations**:
- Pi0.5's Stage 2 (LLM subtask prediction) has different logic from Pi0's Stage 2. Pi0 has no autoregressive text generation; the LLM does a single forward pass. Handle the Pi0.5 path (`pi05=True`) first; add Pi0 support later as needed.
- `past_key_values` structure must be passed through as-is — no cloning or reshaping. This is the most likely source of numerical bugs.

**Deliverable**: Modified `pi0_pytorch.py` that passes numerical consistency tests.

---

### Step 2: Timing System

**Goal**: Implement `SystemTimer` to provide infrastructure for all subsequent performance quantification.

**Tasks**:

2.1. Implement `src/openpi/cache/timing.py`:
  - `SystemTimer` class with context manager `with timer.measure("name"):`
  - GPU timing via `torch.cuda.Event`, CPU timing via `time.perf_counter_ns`
  - Ring buffer storage, `summary()` outputs mean/p50/p95/p99
  - `export_csv()` for raw record export

2.2. Integrate the timer into the three stages from Step 1, replacing Step 0's manual timing. Verify numbers are consistent.

**Why this must be completed before cache logic**: Every subsequent development step needs latency quantification. Without a timer, questions like "how long did cache retrieval take?" and "is it worth it?" cannot be answered.

**Deliverable**: `timing.py` + timing integrated into stage calls.

---

### Step 3: Cache Data Structures and Storage Layer

**Goal**: Implement the cache "warehouse" — able to store, search, and retrieve. This step does NOT involve integration with the inference pipeline.

**Tasks**:

3.1. Implement `src/openpi/cache/storage/cache_entry.py`:
  - `CacheEntry` dataclass (query_key, action_chunk, checkpoint_id, metadata)
  - `CacheResult` dataclass (hit/miss, hit_type, cached_action, cached_noisy_action, cached_timestep)
  - `CheckpointID` enum (CP1, CP2, CP3)
  - `HitType` enum (FULL, WARM_START, PREDICTIVE)

3.2. Implement `src/openpi/cache/storage/vector_store.py`:
  - **CPU-only version for this step**, using FAISS `IndexFlatIP` (inner product; with L2 normalization, equivalent to cosine similarity)
  - `insert(vector, payload)` -> id
  - `search(query, top_k)` -> list of (id, similarity, payload)
  - `delete(id)`
  - `size()`, `clear()`
  - Unit test: insert 1000 random vectors, verify search returns correct top-k

3.3. Implement `src/openpi/cache/config.py`:
  - `CacheConfig` dataclass with all tunable parameters
  - Provide `default_config()` and `debug_config()` factory methods

**This step is independent of inference**: The storage layer is a pure data structure that can be developed and tested standalone. No model loading or GPU required.

**Deliverable**: Data structures + CPU VectorStore + passing unit tests.

---

### Step 4: Orchestrator Skeleton + Interceptor

**Goal**: Connect cache check logic to the inference pipeline, achieving an end-to-end cache workflow. This step uses the simplest component implementations (PlaceholderKeyBuilder + AlwaysSearchGate + ThresholdJudge), with **only CP2 enabled**.

**Why CP2 first, not CP1**:
- CP2 is after the LLM, with subtask information available. The cache semantics are clearest ("same scene + same command -> same action").
- CP2 hits skip flow matching (pure numerical computation), not involving semantic understanding. Lowest risk.
- CP1 skips subtask prediction — high risk, not suitable as the first validation point.
- CP3 is predictive with more complex logic, not suitable for the first round.

**Tasks**:

4.1. Implement `src/openpi/cache/components/key_builder.py`:
  - `QueryKeyBuilder` Protocol
  - `PlaceholderKeyBuilder`: Directly use `stage2_output.command_embedding` with mean pool + L2 normalize, or more simply concatenate `raw_state` with `command_embedding` and normalize. Exact dimensions depend on tensor shapes recorded in Step 0.

4.2. Implement `src/openpi/cache/components/gate.py`:
  - `GateFunction` Protocol
  - `AlwaysSearchGate`: Always returns True

4.3. Implement `src/openpi/cache/components/judge.py`:
  - `SimilarityJudge` Protocol
  - `ThresholdJudge`: cosine similarity > threshold -> hit

4.4. Implement `src/openpi/cache/orchestrator.py`:
  - `CacheOrchestrator`: Combines key_builder + gate + judge + storage
  - `check()` method: gate -> build key -> search -> judge
  - `write_async()` method: Use synchronous writes initially (async deferred to Step 8)

4.5. Implement `src/openpi/cache/interceptor.py`:
  - `InferenceInterceptor`: Wraps Policy, injecting cache checks between stages
  - This step only injects CP2 check after Stage 2

4.6. **End-to-end tests**:
  - Load model, run 10 identical inputs -> 1st should miss, 2nd-10th should hit (identical input)
  - Run 10 different inputs -> all miss
  - Verify that the action returned on hit has L2 distance = 0 from normal inference result (since input is identical)

**Deliverable**: A working end-to-end cache system (CP2 only) that passes the above tests.

---

### Step 5: Core Experiment — Cache Feasibility Validation

**Goal**: Answer the critical question — "For similar but not identical inputs, what is the quality of cache-hit actions?" This determines whether the entire cache system makes sense.

> **This is the first critical experiment milestone for the entire project.** If experiments show that similar inputs produce widely different actions, the cache approach needs fundamental reevaluation. Do not invest further development effort before this experiment.

**Experiment design**:

5.1. **Data preparation**: Collect inference episodes (100-500 steps), recording for each step:
  - Input observation (images, state, prompt)
  - Stage 1 output (vision embedding)
  - Stage 2 output (command embedding)
  - Final action chunk
  - Save all data to disk (HDF5 or pickle)

5.2. **Experiment A: Action continuity in state space**
  - For recorded episodes, compute pairwise state cosine similarity across all steps
  - Compute corresponding action L2 distance
  - Plot scatter: x=state_similarity, y=action_distance
  - **Expected result**: Action distance is low when state similarity is high (positive correlation)
  - **If this trend is not observed**: The cache approach has a fundamental problem

5.3. **Experiment B: Cache hit action quality**
  - Using the Step 4 system, gradually lower the CP2 threshold (from 0.99 to 0.80)
  - Record at each threshold: hit rate, action L2 error (vs normal inference), latency savings
  - Plot three curves: threshold vs hit_rate, threshold vs action_error, threshold vs latency_saving
  - **Find the sweet spot**: Maximize hit rate subject to action error being acceptable (< some value)

5.4. **Experiment C: Discriminative power of different query keys**
  - Compare retrieval quality across several key construction approaches:
    - (a) raw state vector only
    - (b) command embedding only
    - (c) state + command concatenation
    - (d) vision embedding mean pool
  - Metric: precision@k (proportion of top-k retrieved entries whose action is truly close to the current inference action)
  - **This experiment guides subsequent QueryKeyBuilder design**

**Deliverable**: Experiment report with the above plots and conclusions. Decision on whether to continue, plus initial threshold range and key builder direction.

---

### Step 6: CP1 and CP3 Implementation

**Prerequisite**: Step 5 experiment results are positive (cache feasibility validated).

**Tasks**:

6.1. **CP1 implementation**:
  - Inject cache check after Stage 1
  - Key builder needs to handle vision embedding (Step 5 Experiment C will provide direction)
  - CP1 uses a stricter threshold (default 0.98)
  - Test: same scene + same prompt should hit; changing objects or prompt should miss

6.2. **CP3 implementation**:
  - Inject check after Stage 3
  - `schedule_next_action()` mechanism: maintain a `_next_action_scheduled` slot in the orchestrator
  - `should_skip_inference()`: Check at the start of each cycle for a pre-scheduled action
  - CP3 key needs to include action chunk information (since it predicts "the next step")
  - Maintain **consecutive action sequence mapping** — add `next_entry_id` field to entries, pointing to the temporally next entry

6.3. **CP3 special consideration**: CP3 cache entries need to record the "current action -> next action" mapping. This means:
  - When writing to cache, the `next_action_chunk` field cannot be filled until the **next** cycle's action is produced
  - Implement a `DeferredWriter`: Write the entry in cycle N (without next), backfill `next_action_chunk` in cycle N+1

6.4. **Experiments**:
  - Measure CP1/CP2/CP3 hit rates across episodes
  - Quantify latency savings at each checkpoint on hit
  - CP3 predictive accuracy: L2 distance between pre-scheduled action and actually inferred action

**Deliverable**: Complete three-checkpoint system + hit rate and latency reports for each checkpoint.

---

### Step 7: Flow Matching Warm Start

**Prerequisite**: Step 6 complete, CP2 full hit path validated.

**Tasks**:

7.1. Modify `run_stage3()`:
  - When `return_intermediates=True`, save `x_t` at selected timesteps during the flow matching loop
  - Default: save at t=0.7, 0.5, 0.3 (configurable)
  - Saved tensor shape = `[B, action_horizon, action_dim]`, same as noise

7.2. Add `run_stage3_from(stage2_output, start_x, start_t)`:
  - Execute remaining Euler steps starting from `start_x` and `start_t`
  - For example, `start_t=0.3` runs only 3 steps (0.3 -> 0.2 -> 0.1 -> 0.0) instead of 10

7.3. Add warm start judgment logic to the CP2 judge:
  - similarity > `cp2_full_threshold` -> FULL hit
  - `cp2_warm_threshold` < similarity < `cp2_full_threshold` -> WARM_START hit
  - similarity < `cp2_warm_threshold` -> miss

7.4. **Critical experiment: Warm start accuracy vs speed tradeoff**

  This is the second critical experiment milestone.

  - For the same set of inputs, run:
    - (a) Full 10-step flow matching (baseline)
    - (b) Warm start from own cached x_0.7 (3 steps skipped)
    - (c) Warm start from own cached x_0.5 (5 steps skipped)
    - (d) Warm start from own cached x_0.3 (7 steps skipped)
  - Measure action L2 error vs baseline
  - "Own cached" means using intermediate states from **identical input**, isolating warm start error alone (no state similarity concerns)

  - Then warm start with cached x_t from **similar but different** inputs:
    - Take cached x_0.5 from a similar state in the episode
    - Continue denoising with the current observation's velocity field
    - Measure action L2 error
  - **Expected**: Error is within acceptable range and smaller than "directly using cached action without flow matching"

  - Plot trade-off: x=steps_skipped, y=action_error, multiple lines for different state similarity levels

**Deliverable**: Warm start implementation + trade-off experiment data. Determine default warm start timestep and threshold.

---

### Step 8: System Efficiency Optimization — Async and Hardware

**Prerequisite**: Step 7 complete, functional correctness validated. Performance optimization is deferred to this step because timing data from real operation is needed to guide where to invest effort.

**Optimization decision process**: First generate a complete latency breakdown report using Step 2's timer, identify bottlenecks, then optimize in a targeted manner. Do not optimize based on guesses.

**Potential optimization directions** (ordered by expected impact):

8.1. **Async cache writes** (almost certainly needed):
  - Step 4's writes are synchronous, blocking inference
  - Implement `AsyncWriteWorker`: Background thread consuming write requests from a queue
  - Use `threading.Thread` + `queue.Queue` (no need for multiprocessing since writes are I/O bound)
  - Verify: Write latency disappears from the inference critical path

8.2. **GPU VectorStore** (if CPU search is a bottleneck):
  - Check `cp*_search` latency in the timer report
  - If CPU FAISS search latency > 1ms and cache entries > 10k, consider GPU partition
  - Implement `torch.mm` cosine similarity search on a dedicated CUDA stream
  - Use a separate stream to avoid blocking the main inference stream
  - Verify: Search latency drops, main stream inference latency unaffected

8.3. **CUDA Stream isolation** (if cache operations block inference):
  - Implement `CacheHardwareManager`
  - All GPU cache operations (search, key building projections) run on `cache_stream`
  - Pinned memory pool for CPU<->GPU data transfers

8.4. **Gate optimization** (if cache checks themselves become a bottleneck):
  - Implement `StateChangeGate`: Skip search when state change is below threshold
  - Implement `IntervalGate`: Search only every N inference cycles
  - This can dramatically reduce search frequency, especially useful when hit rate is low

8.5. **Eviction strategy** (if cache growth slows down search):
  - Implement `CompositeEviction` (LRU + LFU + quality)
  - Set capacity caps, periodic eviction
  - Eviction runs on a background thread

**Deliverable**: Optimized system + before/after latency comparison report.

---

### Step 9: Query Key Research (Experiment-Intensive)

**Prerequisite**: Step 8 complete, system performance at an acceptable level. A stable running cache system now serves as the experiment platform.

**Why placed here and not earlier**: Query key research needs to be done on a real running system, requiring real hit/miss data and real latency numbers. Researching keys before the system is operational is building castles in the air.

**Experiment directions**:

9.1. **Key information source ablation experiment**:
  - Using collected episode data, compare retrieval quality across different information combinations as keys:

  | Key Combination | Dimension | Precision@5 | Recall@5 | Compute Cost |
  |----------------|-----------|-------------|----------|-------------|
  | raw_state | 32 | ? | ? | Minimal |
  | state + prompt_hash | 32+64 | ? | ? | Low |
  | vision_emb (mean pool) | 2048 | ? | ? | Medium |
  | command_emb (mean pool) | 2048 | ? | ? | Medium |
  | state + command_emb | 32+2048 | ? | ? | Medium |
  | learned projection | 128/256/512 | ? | ? | Requires training |

  - "Precision@5" defined as: proportion of top-5 retrieved entries whose action L2 distance from the current inference action is < epsilon

9.2. **Learned Key Builder** (if simple approaches are insufficient):
  - Train a small projection head (2-3 layer MLP), input is stage output concatenation, output is a low-dimensional key
  - Training objective: contrastive loss — similar states produce close keys, different states produce distant keys
  - Training data from offline episode collection
  - Constraint: Projection head inference latency < 0.5ms (otherwise not worth using cache)

9.3. **Optimal key may differ per checkpoint**:
  - CP1's key lacks command information, may need more vision information
  - CP2's key has command, state information weight can potentially be reduced
  - CP3's key needs action information to predict subsequent actions
  - Tune each checkpoint independently

**Deliverable**: Optimal key builder approach per checkpoint + supporting experiment data.

---

### Step 10: Offline Pre-fill Pipeline

**Prerequisite**: Step 9 complete, key builder approach finalized.

**Tasks**:

10.1. Implement `scripts/prefill_cache.py`:
  - Load episodes from training data or offline rollouts
  - For each timestep, run model inference (or read directly from saved episode data)
  - Build keys, write to VectorStore
  - Support incremental pre-fill (check existing entries, avoid duplicates)

10.2. Serialization/deserialization:
  - VectorStore supports `save(path)` and `load(path)`
  - Save pre-filled cache to disk; load directly at inference time to avoid cold starts

10.3. **Pre-fill quality validation**:
  - Run inference episodes with pre-filled cache
  - Compare action trajectories with cache vs without cache
  - Quantify pre-filled cache hit rate (expected to be much higher than online accumulation from empty cache)

**Deliverable**: Offline pre-fill script + serialization support.

---

### Step 11: Integration Testing and Robustness

**Tasks**:

11.1. **Long-running stability test**:
  - Run 1000+ step episodes, monitoring:
    - Memory leaks (whether cache growth is controlled)
    - Latency stability (whether there's a gradual slowdown trend)
    - Whether hit rate changes reasonably

11.2. **Edge case testing**:
  - Behavior when cache is empty (all miss, normal inference)
  - Eviction behavior when cache is full
  - No crashes on abnormal input (all-black images, empty prompt)
  - Graceful fallback to CPU-only on GPU OOM

11.3. **A/B comparison framework**:
  - Implement `--cache_enabled` flag for one-click cache on/off
  - Compare under identical episodes with cache on/off:
    - End-to-end latency (mean, p95)
    - Action quality (L2 distance from no-cache baseline)
    - GPU utilization

**Deliverable**: Stability test report + A/B comparison data.

---

### Step 12: Advanced Features (On Demand)

The following items are mutually independent; implement based on priority:

12.1. **Metadata DB (MongoDB/SQLite)**:
  - Introduce when rich information beyond vector DB is needed (task name, episode id, success/failure label, etc.)
  - Used for cache analysis and offline quality evaluation
  - Not on the critical path; does not affect inference latency

12.2. **GPU/CPU dynamic migration**:
  - Promotion/Demotion strategies
  - Automatic data tiering based on access frequency

12.3. **Learned Gate Function**:
  - Train a binary classifier to predict "is the current state likely to cache hit?"
  - Input: state delta (vs last), task prompt hash, cache statistics
  - Reduces unnecessary search overhead

12.4. **Distributed Cache**:
  - Multi-robot shared vector DB
  - Requires consideration of network latency and consistency

12.5. **Cache-aware Training**:
  - Introduce cache-hit simulation during training so the model adapts to occasional step-skipping
  - Long-term direction requiring extensive experiments

---

### Development Dependency Graph

```
Step 0: Understand inference pipeline
  │
  ▼
Step 1: Split sample_actions()  ───────────────────────────────────┐
  │                                                                │
  ▼                                                                │
Step 2: Timing system                                              │
  │                                                                │
  ├──────────────────┐                                             │
  ▼                  ▼                                             │
Step 3: Data structs (parallel dev)                                │
  │                                                                │
  ▼                                                                │
Step 4: Orchestrator + Interceptor (CP2 only) ◄────────────────────┘
  │
  ▼
Step 5: ★ Feasibility experiment ★  ── if failed ──> Reevaluate approach
  │ (passed)
  ▼
Step 6: CP1 + CP3 implementation
  │
  ▼
Step 7: Warm start implementation + ★ Accuracy experiment ★
  │
  ▼
Step 8: System efficiency optimization (async, GPU, stream)
  │
  ▼
Step 9: ★ Query key research ★ (experiment-intensive)
  │
  ▼
Step 10: Offline pre-fill
  │
  ▼
Step 11: Integration testing
  │
  ▼
Step 12: Advanced features (on demand)
```

**★-marked steps are critical experiment milestones** whose conclusions directly determine the direction of subsequent work, or even whether to continue.

---

### Estimated Effort Per Step

| Step | Work Type | Primary Deliverable | Estimated Effort |
|------|-----------|---------------------|------------------|
| 0 | Reading + analysis | Baseline report | 1-2 days |
| 1 | Code refactoring | Split pi0_pytorch.py | 2-3 days |
| 2 | Infrastructure dev | timing.py | 1 day |
| 3 | Infrastructure dev | Data structures + VectorStore | 2 days |
| 4 | Core development | Orchestrator + Interceptor | 3-4 days |
| 5 | **Experiment** | Feasibility report | 3-5 days |
| 6 | Core development | CP1 + CP3 | 3-4 days |
| 7 | Dev + **Experiment** | Warm start + tradeoff data | 4-5 days |
| 8 | Performance optimization | Async/GPU/Stream | 3-5 days |
| 9 | **Experiment** | Key builder research | 5-7 days |
| 10 | Tooling dev | Pre-fill script | 2-3 days |
| 11 | Testing | Stability + A/B report | 2-3 days |
| 12 | Advanced | On demand | TBD |

> **Note**: The above time estimates do not include time waiting for experiments to complete or analyzing results. Actual experiment turnaround depends on GPU resources and episode data volume. Steps 2 and 3 have no dependency on each other and can be developed in parallel.