---
name: mi350-kernel-backend-triage
description: Determine whether Triton, CK, or ASM backends can beat existing rocBLAS/hipBLASLt Cijk_* GEMM kernels in OpenPI policy inference on MI350 (gfx950). Use when the user asks if Triton/CK/ASM can be faster than Cijk, asks to replace Cijk kernels, requests ranking by occurrence/total time, or wants a workflow to microbench + validate + integrate kernel choices without FP8.
---

# MI350 kernel backend triage (Cijk vs Triton/CK/ASM)

## Goal
Answer (with data): **can Triton, CK, or ASM beat the current `Cijk_*` kernels end-to-end?**

This skill enforces:
- **Occurrence-aware ranking**: always optimize by **total time** (sum over calls), not “single kernel looks big”.
- **Two-tier evidence**: microbench (per-call) **and** end-to-end (replay makespan).
- **Numeric correctness guardrails**: no “faster but wrong” unless explicitly asked.
- **No FP8**: keep BF16 path unless user explicitly changes.

## Default setup (OpenPI policy inference)
Use the repo’s policy inference benchmark and target GPU7 unless the environment masks devices.

Common env knobs:
- `OPENPI_GPU_ID=7`
- `OPENPI_MANUAL_CUDAGRAPH=1` (graph replay path)
- `OPENPI_TIMING=cuda_event`
- `AITER_PRESHUFFLE_WEIGHTS=0/1`
- `AITER_CONFIG_GEMM_BF16=/sgl-workspace/openpi/traces/gemm_tuning/openpi_bf16_tuned_gemm.csv` (when comparing tuned GEMM behavior)

## Step 1: Rank by total time (include occurrence)
When a trace/rocprof output exists, build a table:
- `kernel_name`
- `calls`
- `total_ms` (sum of durations across calls)
- `avg_us` (\(total\_ms / calls\))
- `pct_of_replay`

If using `rocprofv3`, prefer isolating steady-state with markers:
1) Enable ROCTX-like ranges in the benchmark (`OPENPI_ROCTX=1`).
2) Run:

```bash
OPENPI_GPU_ID=7 OPENPI_ROCTX=1 OPENPI_MANUAL_CUDAGRAPH=1 OPENPI_TIMING=cuda_event \
rocprofv3 --kernel-trace 1 --memory-copy-trace 1 --marker-trace 1 --output-format csv -d <outdir> -- \
python scripts/benchmark_policy_inference.py
```

3) Filter events inside `openpi/replay_iter_*` and aggregate by kernel name.

## Step 2: Map hot `Cijk_*` kernels → GEMM problem shapes
For each hot `Cijk_*` family, identify its GEMM(s):
- Extract shapes from existing OpenPI trace analysis (`scripts/analyze_policy_trace.py`) and/or from AITER gemm shape logs (`openpi_bf16_untuned_gemm.csv`).
- Track whether it’s bias/no-bias, and whether it’s part of attention/MLP/projections.

The goal is to know which shapes you must beat **and how many times per iteration**.

## Step 3: Microbench candidate backends (Triton / CK / ASM) on the SAME shapes
Use the existing harness:
- `scripts/unit_test_aiter_solutions.py` (GEMM backends)
- `scripts/unit_test_aiter_attention.py` (attention)

Rules:
- Use the same dtype/layouts as the real run (BF16, same strides/contiguity).
- Warm up sufficiently.
- Measure **median** time per call.
- Record: backend, config/solidx, and achieved time.

If a backend is faster per call, estimate potential e2e gain:
\[
\Delta ms \approx calls\_per\_iter \times (t_{old} - t_{new})
\]
Only proceed if the estimated \(\Delta ms\) is meaningful (e.g. \(\ge 0.3\) ms) after accounting for integration overhead.

## Step 4: Integrate the winner with minimal blast radius
Preferred integration order:
1) **AITER tuned CSV** (swap solution for that exact `(M,N,K,bias)`); lowest risk.
2) **Targeted dispatch heuristic** (only for known hot shapes); keep an env gate.
3) **New fused kernel** (highest risk; only if it reduces kernel count or removes heavy transposes).

Avoid changing many shapes at once; do one candidate at a time to preserve attribution.

## Step 5: Validate end-to-end, not just microbench
For every candidate change:
- Run policy inference in graph replay mode.
- Collect:
  - `OPENPI_TIMING=cuda_event` P50/P95
  - One replay trace (profiler) OR rocprof marker window
- Re-rank by total time again.

Success criteria:
- Total replay makespan goes down (not just one kernel bucket).
- No large regressions in other kernels (layout/transpose/fusion fallout).

## Step 6: Numeric correctness
Run numeric check when integrating kernel changes:

```bash
OPENPI_NUMERIC_CHECK=1 OPENPI_NUMERIC_STRICT=1 OPENPI_MANUAL_CUDAGRAPH=1 \
python scripts/benchmark_policy_inference.py
```

If the user explicitly asks for “even if wrong, show latency”, allow `OPENPI_NUMERIC_STRICT=0` but still fail on NaNs.

## Common pitfalls (why “ASM/CK faster” doesn’t win e2e)
- **Different layouts**: extra transposes/contiguity conversions erase wins.
- **Bias/epilogue mismatch**: faster no-bias kernel becomes slower once bias is fused or split incorrectly.
- **Kernel count increases**: many small kernels inflate makespan even if each is fast.
- **Autotune mismatch**: microbench picks a solution that interacts poorly with real scheduling/occupancy.
- **Dispatch overhead**: Python-side or dynamic shape dispatch can dominate if called hundreds of times.

## Output format (always)
When answering “can Triton/CK/ASM beat Cijk?”:
- Provide a ranked table by **total_ms** for replay.
- For each target kernel family, provide:
  - `calls`, `total_ms`, `avg_us`
  - best-known alternative backend microbench time
  - estimated and measured e2e delta
  - whether it passed numeric check
