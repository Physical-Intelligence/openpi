## MI350 (gfx950) OpenPI policy inference — hot `Cijk_*` GEMMs (shareable)

### What this is
This is a **shareable, occurrence-aware** snapshot of the hottest hipBLASLt/rocBLAS GEMM kernels (`Cijk_*`) inside the **steady-state replay iteration** of OpenPI policy inference (Pi0).

The goal is to help a colleague answer:
- “Which GEMMs dominate?”
- “What are the exact shapes?”
- “What kernel (tiling) was chosen?”
- “How many times per iteration?”
- “What’s the measured total time impact?”

### Measurement method (ground truth)
Source: `rocprofv3` **inside a ROCTX marker window** around the replay iteration:
- Marker: `openpi/replay_iter_1`
- Marker duration: **33.672929 ms**
- Inside marker: **3743 kernel dispatches**, **0 memcpy** (steady-state)

Raw data directory:
- `openpi/traces/rocprofv3_comp_30_2_marker2/smci350-zts-gtu-f16-25/`
  - `669991_kernel_trace.csv`
  - `669991_kernel_stats.csv`
  - `669991_marker_api_trace.csv`

### Run configuration (repro)
This run used:
- `OPENPI_GPU_ID=7`
- `OPENPI_MANUAL_CUDAGRAPH=1`
- `OPENPI_TIMING=cuda_event`
- `AITER_PRESHUFFLE_WEIGHTS=0`
- `AITER_CONFIG_GEMM_BF16=/sgl-workspace/openpi/traces/gemm_tuning/openpi_bf16_tuned_gemm.csv`
- `OPENPI_ROCTX=1` (to create marker ranges)

Repro command (writes a new `rocprofv3` capture dir):

```bash
rm -rf /sgl-workspace/openpi/traces/rocprofv3_comp_30_2_marker2 && \
mkdir -p /sgl-workspace/openpi/traces/rocprofv3_comp_30_2_marker2 && \
OPENPI_GPU_ID=7 OPENPI_ROCTX=1 \
AITER_PRESHUFFLE_WEIGHTS=0 \
AITER_CONFIG_GEMM_BF16=/sgl-workspace/openpi/traces/gemm_tuning/openpi_bf16_tuned_gemm.csv \
OPENPI_MANUAL_CUDAGRAPH=1 OPENPI_TIMING=cuda_event \
WARMUP=1 ITERATIONS=1 NUM_STEPS=10 \
rocprofv3 --kernel-trace 1 --memory-copy-trace 1 --marker-trace 1 --stats 1 --output-format csv \
  -d /sgl-workspace/openpi/traces/rocprofv3_comp_30_2_marker2 -- \
  python /sgl-workspace/openpi/scripts/benchmark_policy_inference.py
```

### Top `Cijk_*` kernels by **total time** (occurrence-aware)
All numbers below are **within `openpi/replay_iter_1`**.

Legend:
- \(M,N,K\) correspond to GEMM \(C[M,N] = A[M,K] @ B^T[N,K]\)
- “bias” means an epilogue bias add is fused in the GEMM kernel

| Rank | Kernel (hipBLASLt name prefix) | Calls / iter | Total ms / iter | Avg µs / call | % of replay | GEMM (M,N,K,bias) | A / W / out shapes |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | `Cijk_...MT32x64x128...` | 324 | 2.849 | 8.79 | 8.52% | **(256,1152,1152, bias=True)** | A=[256,1152], W=[1152,1152], bias=[1152], out=[256,1152] |
| 2 | `Cijk_...MT256x16x64...` | 180 | 2.833 | 15.74 | 8.47% | **UNKNOWN (not logged yet)** | UNKNOWN |
| 3 | `Cijk_...MT16x16x128...` | 360 | 2.685 | 7.46 | 8.03% | **UNKNOWN (not logged yet)** | UNKNOWN |
| 4 | `Cijk_...MT16x16x512...` | 360 | 2.609 | 7.25 | 7.80% | **Likely small-M GEMMs** (see below) | likely A=[11,K], W=[1024,K], out=[11,1024] |

These four sum to **10.976 ms (32.8% of replay makespan)**.

### Known shape ↔ chosen kernel mapping (from tuned-GEMM logs)
The file below records exact shapes and the chosen hipBLASLt kernel name:
- `openpi/traces/gemm_tuning/openpi_bf16_tuned_gemm.csv`

Entries (examples relevant to the hot kernels):
- **(256,1152,1152,bias=True)** → `Cijk_...MT32x64x128...`
- Small-M shapes observed using `Cijk_...MT16x16x512...`:
  - **(11,1024,2048,bias=False)** → `Cijk_...MT16x16x512...`
  - **(11,1024,4096,bias=False)** → `Cijk_...MT16x16x512...`

### Optional: dump hipBLASLt GEMM shape ↔ kernelName mapping (programmatic)
If you want an easy “what (M,N,K) maps to which `Cijk_*` kernelName?” export without digging through traces:

```bash
OPENPI_LOG_HIPBLASLT_GEMMS=1 \
OPENPI_HIPBLASLT_GEMM_LOG=/sgl-workspace/openpi/traces/hipblaslt_gemm_shapes.csv \
python /sgl-workspace/openpi/scripts/benchmark_policy_inference.py
```

This writes:
- `openpi/traces/hipblaslt_gemm_shapes.csv` (columns: `kernelName,M,N,K,bias,calls`)

Note: `calls` in that CSV are **run-level counts** (warmup + capture + benchmark), not “per replay iteration”.

### What’s missing (and how to get it quickly)
Two of the top kernels (`MT256x16x64` and `MT16x16x128`) are **not currently present in** `openpi_bf16_tuned_gemm.csv`, so their exact `(M,N,K,bias)` mapping is not written down in a single place yet.

To resolve this for colleagues who want to propose CK/ASM alternatives, the next step is to add **one explicit GEMM-shape logging hook** for the remaining matmul paths (not going through the aiter tuned-GEMM logger), so we can produce:
- `(M,N,K,bias)` ↔ `Cijk_*` kernel name ↔ `calls` ↔ `total_ms`

### “Can CK/ASM replace these?”
For these top `Cijk_*` kernels:
- Any replacement must win on **total_ms = calls × avg** *and* avoid extra packing/transposes.
- We already saw cases where “faster in microbench” (different `solidx` / ASM preshuffle) **regressed end-to-end** due to integration effects.

If a colleague has a CK candidate, the actionable target is:
- Beat the per-call time for the exact `(M,N,K,bias)` above,
- then validate e2e by re-running this marker-based table and checking replay makespan improves.

