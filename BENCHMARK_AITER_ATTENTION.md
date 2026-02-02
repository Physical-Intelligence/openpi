# AMD MI350 Benchmark - OpenPI Pi0 (3.5B)

## Full Policy Inference (batch=1)

```bash
python scripts/benchmark_policy_inference.py
```

| Metric | Value |
|--------|-------|
| Latency | **(stale; see below)** |
| Throughput | **(stale; see below)** |
| Memory | 7.10 GB |

**Important**: the 57.6ms / 17.36Hz numbers were measured under an older benchmark setup and are **not representative** of the current best-known MI350 path.

Current best-known MI350 policy inference uses:
- manual full-call graph replay: `OPENPI_MANUAL_CUDAGRAPH=1`
- event timing: `OPENPI_TIMING=cuda_event`
- allow compiling through aiter attention: `OPENPI_DISABLE_COMPILE_AITER_ATTN=0`
- disable inductor memory planning on ROCm: `OPENPI_INDUCTOR_MEMORY_PLANNING=0`
- tuned GEMM config: `AITER_CONFIG_GEMM_BF16=openpi/traces/gemm_tuning/openpi_bf16_tuned_gemm.csv`

Repro (MI350, GPU7):

```bash
OPENPI_GPU_ID=7 \
OPENPI_MANUAL_CUDAGRAPH=1 OPENPI_TIMING=cuda_event \
OPENPI_DISABLE_COMPILE_AITER_ATTN=0 OPENPI_INDUCTOR_MEMORY_PLANNING=0 \
AITER_PRESHUFFLE_WEIGHTS=0 \
AITER_CONFIG_GEMM_BF16=/sgl-workspace/openpi/traces/gemm_tuning/openpi_bf16_tuned_gemm.csv \
WARMUP=3 ITERATIONS=10 NUM_STEPS=10 \
python scripts/benchmark_policy_inference.py
```

## Precision Verification

```bash
python scripts/verify_precision.py
```

| Metric | Value |
|--------|-------|
| Cosine Similarity | **1.000000** |
| Result | **PASSED** |

## Profiling Trace

```bash
python scripts/dump_trace.py
```

View `traces/policy_inference.json` in [Perfetto](https://ui.perfetto.dev/)
