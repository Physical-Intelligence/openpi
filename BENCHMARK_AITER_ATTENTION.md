# AMD MI350 Benchmark - OpenPI Pi0 (3.5B)

## Full Policy Inference (batch=1)

```bash
python scripts/benchmark_policy_inference.py
```

| Metric | Value |
|--------|-------|
| Latency | **57.6 ms** |
| Throughput | **17.36 Hz** |
| Memory | 7.10 GB |

Uses `torch.compile(mode="max-autotune")` for optimal kernel selection.

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
