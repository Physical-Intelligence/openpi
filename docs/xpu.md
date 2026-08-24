## Intel XPU (Arc GPU) Inference

OpenPI supports inference on Intel Arc GPUs (e.g. Arc B70) via the PyTorch XPU backend.
Only the PyTorch model path is supported — JAX inference requires CUDA or CPU.

---

### Prerequisites

1. **Intel GPU driver** — install the [Intel compute runtime](https://github.com/intel/compute-runtime/releases) (version 26.18 or later recommended).

2. **Intel oneAPI Base Toolkit** — required for the SYCL/XPU runtime that PyTorch links against:
   ```bash
   wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
     | gpg --dearmor \
     | sudo tee /usr/share/keyrings/oneapi-archive-keyring.gpg
   echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
     | sudo tee /etc/apt/sources.list.d/oneAPI.list
   sudo apt update && sudo apt install intel-basekit
   ```

---

### 1. Clone and switch to the XPU branch

```bash
git clone --recurse-submodules https://github.com/munikera/openpi.git
cd openpi
git checkout add-intel-xpu-inference
```

---

### 2. Install dependencies

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra xpu
```

This installs `torch==2.11.0+xpu`, `torchvision==0.26.0+xpu`, and `triton-xpu==3.7.0` from
the PyTorch XPU index instead of the default CUDA wheels.

Then apply the required transformers patches (needed for the PyTorch model path):

```bash
cp -r ./src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/
```

> **Note**: you must re-run this `cp` after every `uv sync`, since sync may overwrite the
> patched files. With uv's default hardlink mode the patch also modifies the shared
> transformers cache. To fully undo it later run `uv cache clean transformers`.

---

### 3. Verify XPU is detected

```bash
uv run python -c "
import torch
print('torch version :', torch.__version__)
print('xpu available :', torch.xpu.is_available())
print('xpu device count:', torch.xpu.device_count())
for i in range(torch.xpu.device_count()):
    print(f'  device {i}:', torch.xpu.get_device_properties(i).name)
"
```

Expected output on a B70:
```
torch version : 2.11.0+xpu
xpu available : True
xpu device count: 2
  device 0: Intel(R) Graphics [0xe223]
  device 1: Intel(R) Graphics [0xe223]
```

---

### 4. Download and convert the π₀.₅ model to PyTorch

The XPU backend requires a PyTorch checkpoint. The published π₀.₅-LIBERO checkpoint is JAX,
so download and convert it (one-time, ~10 GB download).

**Step 1 — download the JAX checkpoint** (no GCP credentials required):
```bash
uv run python -c "
from openpi.shared.download import maybe_download
path = maybe_download('gs://openpi-assets/checkpoints/pi05_libero')
print('Downloaded to:', path)
"
```
This caches the checkpoint to `~/.cache/openpi/openpi-assets/checkpoints/pi05_libero`.

**Step 2 — convert to PyTorch:**
```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint-dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \
    --config-name pi05_libero \
    --output-path checkpoints/pi05_libero_pytorch
```

**Step 3 — copy the norm stats** (the convert script only copies model weights):
```bash
cp -r ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets \
    checkpoints/pi05_libero_pytorch/
```

The converted checkpoint is saved to `checkpoints/pi05_libero_pytorch/`.

---

### 5. Smoke test — verify XPU inference end-to-end

**Terminal 1 — start the policy server:**
```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir checkpoints/pi05_libero_pytorch
```

The server will log autotune output for the first connection (one-time JIT compile, takes
1–3 minutes). Wait until you see:
```
INFO:websockets.server:server listening on 0.0.0.0:8000
```
before sending any requests.

> **Note:** The first `infer` call triggers `torch.compile(mode='max-autotune')` and can take
> several minutes. The server disables WebSocket keepalive pings so the client connection stays
> open during compilation.

**Terminal 2 — send LIBERO-shaped dummy observations:**
```bash
uv run examples/simple_client/main.py --port 8000 --env LIBERO --num-steps 5
```

Expected output (verified on B70):
```
Running policy: 100%|██████████| 5/5 [00:00<00:00,  7.6it/s]
  client_infer_ms  ~130 ms
  policy_infer_ms  ~124 ms
```

---

### 6. Run LIBERO benchmark inference on XPU

The LIBERO eval requires its own environment. Initialize the submodule first if you
haven't already:

```bash
git submodule update --init --recursive
```

**Terminal 1 — policy server** (keep running from step 5, or restart):
```bash
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir checkpoints/pi05_libero_pytorch
```

**Terminal 2 — LIBERO client:**
```bash
# Create the LIBERO venv (Python 3.8 required by LIBERO)
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
export LD_LIBRARY_PATH="$(pwd)/examples/libero/.venv/lib:$LD_LIBRARY_PATH"
export MUJOCO_GL=osmesa
export NUMBA_DISABLE_JIT=1

# On kernels with READ_IMPLIES_EXEC hardening, clear the execstack flag from
# the old PyTorch build to avoid "cannot enable executable stack" ImportError:
patchelf --clear-execstack examples/libero/.venv/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so

# Run libero_spatial (default suite: 10 tasks × 50 trials)
python examples/libero/main.py --args.port 8000
```

To run a different task suite:
```bash
python examples/libero/main.py --args.task-suite-name libero_10
```

Available suites: `libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90`.

Expected results (π₀.₅ checkpoint at 30k steps):

| Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|:-:|:-:|:-:|:-:|:-:|
| 98.8% | 98.2% | 98.0% | 92.4% | 96.85% |

---

### Selecting a specific GPU

To target a specific XPU device (e.g. when two are present), set `ZE_AFFINITY_MASK` before
starting the server:

```bash
export ZE_AFFINITY_MASK=0   # use device 0
uv run scripts/serve_policy.py --port 8000 policy:checkpoint \
    --policy.config pi05_libero \
    --policy.dir checkpoints/pi05_libero_pytorch
```

Set `ZE_AFFINITY_MASK=1` to use device 1 instead.

---

### Tested hardware

Two Intel Arc B70 GPUs (`device 0xe223`, `xe` kernel driver, Ubuntu 22.04).
Required VRAM for inference: > 8 GB.
