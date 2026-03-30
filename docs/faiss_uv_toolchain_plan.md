# Faiss GPU Build Plan for `uv` Environment

## Goal

Use a dedicated Conda environment only as a build toolchain, then build and install a GPU-enabled Faiss package into `openpi/.venv` managed by `uv`.

This plan is designed to satisfy these constraints:

- `openpi` main runtime remains in the existing `uv` environment
- No root access is required
- Target is GPU + CPU hybrid vector retrieval
- Avoid direct package copying from Conda into `uv`
- Keep rollback and verification paths explicit

## Scope

This document only covers the following route:

- Conda provides build tools
- Faiss is built from source
- The final Python package is installed into `openpi/.venv`

This document does not cover:

- Running the full vector DB as a separate Conda service
- Installing CPU-only Faiss
- Using archived PyPI `faiss-gpu` wheels
- Using `faiss-gpu-cuvs`

## Current Environment Assumptions

Based on the current repository and local environment:

- `openpi` uses Python `3.11`
- `openpi/.venv` is the main runtime environment
- `torch==2.7.1+cu126`
- `torch.cuda.is_available()` is already `True`
- GPU driver/runtime is already usable from PyTorch
- `nvcc` is not currently available in `PATH`
- `swig` is not currently available in `PATH`

These assumptions imply:

- We do not need to install a GPU driver
- We do need a user-space CUDA build toolchain
- We should not attempt to use PyPI `faiss-gpu`

## Recommended Target

- Faiss version: `1.14.1`
- Build mode: source build with GPU support enabled
- Target install environment: `/home/ziyang10/openpi/.venv`

## High-Level Strategy

1. Create a dedicated Conda build environment.
2. Install CUDA build components and compilation tools into that environment.
3. Download Faiss source code at `v1.14.1`.
4. Configure the build so that Python bindings target `openpi/.venv/bin/python`.
5. Build a wheel or installable artifact for the `uv` environment.
6. Install the artifact into `openpi/.venv`.
7. Validate import, CPU index, and GPU index behavior inside `openpi/.venv`.
8. Only after validation, integrate Faiss into step3 code paths.

## Why This Route

This route is the "second safest" option:

- safer than copying Conda packages into `uv`
- safer than trying to make the whole `openpi` runtime switch to Conda
- less isolated than a separate retrieval service, but still manageable

It works because:

- Conda can provide `nvcc`, CUDA headers, dev libraries, `cmake`, and `swig` without root
- the final build can target the actual Python interpreter used by `openpi`

## Critical Non-Goals

The following approaches are explicitly discouraged:

- copying `faiss` Python packages from Conda `site-packages` into `openpi/.venv`
- copying `.so` files by hand into `openpi/.venv`
- mixing Conda Python and `uv` Python in one process without an explicit build/install target
- using `pip install faiss-gpu` from PyPI for Python 3.11

## Phase 1: Prepare the Conda Build Toolchain

Create a dedicated Conda environment used only for building Faiss.

Suggested contents:

- `python=3.11`
- `cuda-nvcc`
- `cuda-libraries-dev`
- `cuda-cupti`
- `cuda-nvtx`
- `cmake`
- `swig`
- `cxx-compiler`
- `pkg-config`
- `ninja` or `make`
- `openblas` or `libopenblas`
- `git`

### Phase 1 Success Criteria

The build environment should pass all of the following:

- `nvcc --version`
- `swig -version`
- `cmake --version`
- compiler tools are available
- CUDA headers and libraries are visible under the Conda prefix

### Notes

- This Conda environment is a toolchain only.
- It is not intended to replace `openpi/.venv`.
- It should not become the default runtime for `openpi`.

## Phase 2: Create a Clean Build Workspace

Use a dedicated build workspace outside the repository to avoid polluting `openpi`.

Suggested layout:

```text
~/faiss-build/
  faiss-src/
  build/
  wheelhouse/
  logs/
```

Recommended rules:

- keep source, build artifacts, and wheels separate
- do not build Faiss directly inside the `openpi` repo
- keep logs for troubleshooting

## Phase 3: Pin the Faiss Source Version

Clone the Faiss repository and checkout:

- tag `v1.14.1`

Do not build from an arbitrary branch tip. Use a tagged release so that:

- the build is reproducible
- later debugging is easier
- wheel provenance is clear

## Phase 4: Bind the Build to the `uv` Python

This is the most important part of the plan.

The build tools come from Conda, but the Python target must be:

- `/home/ziyang10/openpi/.venv/bin/python`

The build configuration must explicitly point to that interpreter for:

- Python executable
- Python include path
- Python site-packages target

### Why This Matters

If the build accidentally targets Conda's Python instead of `openpi/.venv`, the resulting package will not be a true install target for the `uv` environment.

That usually causes one of these failures:

- install succeeds but import fails
- build succeeds but the package is bound to the wrong ABI
- package only works inside the Conda build environment

### Phase 4 Success Criteria

Build logs should clearly show that the Python executable in use is:

- `/home/ziyang10/openpi/.venv/bin/python`

## Phase 5: Configure the Faiss Build

Build configuration should prioritize the smallest successful GPU-enabled path.

Key settings:

- enable GPU support
- enable Python bindings
- disable unnecessary tests initially
- point BLAS to the Conda-provided implementation
- point CUDA paths to the Conda toolchain environment

Recommended intent:

- build minimal features first
- avoid spending time on optional components before import/search works

## Phase 6: Produce an Installable Artifact

Preferred outcome:

- produce a wheel for installation into `openpi/.venv`

Acceptable fallback:

- direct source installation into `openpi/.venv`

The wheel-first approach is preferred because it is:

- repeatable
- easier to reinstall
- easier to remove
- easier to archive

### Success Criteria

At the end of this phase, there should be an explicit artifact, ideally in:

- `~/faiss-build/wheelhouse/`

## Phase 7: Runtime Library Strategy

Installing the wheel into `openpi/.venv` is not the whole problem.

Runtime must also be able to locate:

- `libfaiss.so`
- `libfaiss_gpu.so`
- CUDA runtime libraries
- BLAS libraries
- any required C++ runtime libraries

### Short-Term Strategy

Use a launch wrapper or environment setup that exports:

- `LD_LIBRARY_PATH`

pointing to the required library directories under the Conda toolchain prefix and any Faiss build output directories.

### Long-Term Strategy

Reduce dependence on `LD_LIBRARY_PATH` by improving link layout or embedding appropriate runtime search paths during build.

### Rule

Do not treat a successful `pip install` as proof that the integration is complete.

## Phase 8: Install into `openpi/.venv`

Install the built artifact into:

- `/home/ziyang10/openpi/.venv`

This step should happen only after:

- the build target was confirmed to be the `uv` Python
- the artifact was successfully produced

Before installation:

- record current environment state if needed
- keep the wheel artifact for rollback and reinstallation

## Phase 9: Minimal Validation in `openpi/.venv`

Validation must happen inside the actual `uv` environment.

### Validation Layer 1: Import

Required checks:

- `import faiss`
- read Faiss version

Goal:

- verify that Python bindings load in the real runtime

### Validation Layer 2: CPU Index

Required checks:

- create a simple `IndexFlatL2`
- add a few vectors
- run a search

Goal:

- verify that base Faiss functionality works

### Validation Layer 3: GPU Index

Required checks:

- initialize GPU resources
- move a CPU index to GPU
- add/search vectors on GPU
- verify basic CPU/GPU transfer behavior if needed

Goal:

- verify that the actual GPU build path is functional in `openpi/.venv`

### Validation Exit Condition

Only proceed to code integration if all three validation layers pass.

## Phase 10: Integrate into Step3 Incrementally

Do not wire Faiss directly into all cache checkpoints at once.

Instead:

1. add a small vector store abstraction
2. validate add/search behavior first
3. integrate one checkpoint path first
4. expand only after retrieval behavior is stable

Suggested first integration point:

- `CP2`

Reason:

- it balances semantic consistency and cache benefit better than an immediate `CP1` first attempt

## Rollback Plan

Rollback must be straightforward.

Required safeguards:

- keep build workspace separate from repo code
- keep wheel artifacts
- avoid direct manual copying of shared libraries into `uv`
- install only after artifact creation and validation prep
- stop if runtime linking becomes too fragile

Rollback options:

- uninstall the built Faiss package from `openpi/.venv`
- clear temporary environment variable injections
- discard build directory without touching repo code

## Go / No-Go Criteria

### Go

Proceed with this approach only if:

- Faiss imports inside `openpi/.venv`
- CPU index operations work
- GPU index operations work
- runtime linking is understandable and repeatable

### No-Go

Stop and switch to a separate retrieval service if:

- the build keeps targeting Conda Python by accident
- `import faiss` works but GPU execution is unstable
- runtime requires brittle or unclear library path hacks
- library conflicts with PyTorch or NumPy become hard to control

## Main Risks

1. Python bindings link against the wrong interpreter.
2. The wheel installs, but runtime shared libraries cannot be resolved.
3. CUDA toolchain and the existing PyTorch CUDA runtime interact poorly.
4. BLAS or C++ runtime dependencies become mixed across environments.

## Practical Recommendation

Use this route only if you have a real reason to keep Faiss in the same Python process as `openpi`.

If repeated runtime-linking problems appear, stop early and switch to:

- `uv` for the main `openpi` runtime
- a separate Conda-based Faiss retrieval process or service

That route is still the safer overall architecture when environment stability is the top priority.

## Executed Build and Install Record

This section records the actual steps executed in this workspace for the current successful Faiss GPU build and equivalent install into `openpi/.venv`.

### Final Outcome

The following is true in the current environment after the steps below:

- Faiss `1.14.1` was built from source with GPU support
- the build targeted `openpi/.venv/bin/python`
- the built Python package was copied into `openpi/.venv` `site-packages`
- `import faiss` works from `openpi/.venv`
- CPU and GPU smoke tests both passed
- after the equivalent install, neither `PYTHONPATH` nor `LD_LIBRARY_PATH` was required for import or GPU smoke testing on this machine

### Paths Used

- Conda build environment:
  - `/home/ziyang10/.conda/envs/faiss-build`
- Faiss source:
  - `/home/ziyang10/faiss-build/faiss-src`
- CMake build directory:
  - `/home/ziyang10/faiss-build/build`
- Installed C++ libraries:
  - `/home/ziyang10/faiss-build/install`
- Target `uv` runtime:
  - `/home/ziyang10/openpi/.venv`
- Final equivalent installed Python package:
  - `/home/ziyang10/openpi/.venv/lib/python3.11/site-packages/faiss`

### Step 1: Create the Conda Toolchain Environment

Executed:

```bash
conda create -y -n faiss-build -c conda-forge python=3.11 cmake swig cxx-compiler pkg-config ninja git openblas
conda install -y -n faiss-build -c nvidia cuda-nvcc cuda-libraries-dev cuda-cupti cuda-nvtx
```

Verified:

- `nvcc --version`
- `swig -version`
- `cmake --version`

Observed toolchain version highlights:

- CUDA compiler: `12.9.86`
- SWIG: `4.4.1`
- CMake: `4.3.1`

### Step 2: Prepare the Faiss Source Tree

Executed:

```bash
mkdir -p /home/ziyang10/faiss-build/{build,wheelhouse,logs}
git clone --branch v1.14.1 --depth 1 https://github.com/facebookresearch/faiss.git /home/ziyang10/faiss-build/faiss-src
```

Resolved source commit:

- `5622e93733b64b2e033362dbdfda019b2ab33ef0`

### Step 3: Bind CMake to the `uv` Python

Important observations from the target runtime:

- Python executable:
  - `/home/ziyang10/openpi/.venv/bin/python`
- Python include directory exposed by that interpreter:
  - `/home/ziyang10/.conda/envs/lerobot/include/python3.11`
- NumPy include directory:
  - `/home/ziyang10/openpi/.venv/lib/python3.11/site-packages/numpy/core/include`

This means the `uv` virtual environment is using a Python interpreter whose headers live under the Conda-provided Python backing that environment. This is expected in the current setup and was not modified during the build.

### Step 4: Run the CMake Configure Step

Executed:

```bash
source /home/ziyang10/miniforge3/etc/profile.d/conda.sh
conda activate faiss-build

cmake -S /home/ziyang10/faiss-build/faiss-src \
  -B /home/ziyang10/faiss-build/build \
  -G Ninja \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_MKL=OFF \
  -DFAISS_ENABLE_EXTRAS=OFF \
  -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_OPT_LEVEL=avx2 \
  -DBLA_VENDOR=OpenBLAS \
  -DCUDAToolkit_ROOT=$CONDA_PREFIX \
  -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DPython_EXECUTABLE=/home/ziyang10/openpi/.venv/bin/python \
  -DPython_ROOT_DIR=/home/ziyang10/openpi/.venv \
  -DPython_INCLUDE_DIR=/home/ziyang10/.conda/envs/lerobot/include/python3.11 \
  -DPython_NumPy_INCLUDE_DIR=/home/ziyang10/openpi/.venv/lib/python3.11/site-packages/numpy/core/include
```

Configuration succeeded.

Key cache values confirmed:

- `FAISS_ENABLE_GPU=ON`
- `FAISS_ENABLE_PYTHON=ON`
- `FAISS_ENABLE_MKL=OFF`
- `BLA_VENDOR=OpenBLAS`
- `CUDAToolkit_ROOT=/home/ziyang10/.conda/envs/faiss-build`
- `Python_EXECUTABLE=/home/ziyang10/openpi/.venv/bin/python`
- `CMAKE_CUDA_ARCHITECTURES=90`

### Step 5: Compile Faiss

The manual compile command used after configuration was:

```bash
cmake --build /home/ziyang10/faiss-build/build -j 8 --target faiss faiss_avx2 swigfaiss swigfaiss_avx2
```

Observed during the build:

- repeated NVCC warnings such as `warning #611-D`
- these warnings were about partially overridden overloaded virtual functions
- they did not stop the build
- they were treated as normal warnings, not build failures

The build completed successfully.

### Step 6: Install the Built C++ Libraries

Executed:

```bash
cmake --install /home/ziyang10/faiss-build/build --prefix /home/ziyang10/faiss-build/install
```

This installed:

- `libfaiss.so`
- `libfaiss_avx2.so`
- Faiss headers
- Faiss CMake package files

under:

- `/home/ziyang10/faiss-build/install`

### Step 7: Initial `LD_LIBRARY_PATH` Setup Attempt

An initial attempt to export `LD_LIBRARY_PATH` failed because the shell line was split incorrectly by the terminal.

Broken example pattern:

```bash
export LD_LIBRARY_PATH=/some/path:/another/path:/split
  path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

This caused a shell error because the newline broke the command into two pieces.

The corrected one-line form was:

```bash
export LD_LIBRARY_PATH=/home/ziyang10/faiss-build/install/lib:/home/ziyang10/.conda/envs/faiss-build/lib:/home/ziyang10/.conda/envs/faiss-build/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

### Step 8: Generate the Python Package Build Output

At this point, `faiss/python` contained generated `.so` files and `setup.py`, but it was not yet importable as a complete package from the target interpreter.

Executed:

```bash
cd /home/ziyang10/faiss-build/build/faiss/python
/home/ziyang10/openpi/.venv/bin/python setup.py build
```

This created:

- `/home/ziyang10/faiss-build/build/faiss/python/build/lib/faiss`

which is a complete Python package directory ready for import or equivalent local installation.

### Step 9: Validate Before Installing into `site-packages`

Before copying into `site-packages`, the package was tested with:

- `PYTHONPATH=/home/ziyang10/faiss-build/build/faiss/python/build/lib`
- `LD_LIBRARY_PATH` pointing to the installed Faiss and Conda CUDA libraries

Validation passed:

- `import faiss`
- CPU smoke test
- GPU smoke test

### Step 10: Equivalent Install into `openpi/.venv`

Since `openpi/.venv` did not have `pip` available and no `faiss` package already existed in `site-packages`, the equivalent local install used a direct copy of the built package directory.

Executed:

```bash
cp -r /home/ziyang10/faiss-build/build/faiss/python/build/lib/faiss /home/ziyang10/openpi/.venv/lib/python3.11/site-packages/
```

This made Faiss available at:

- `/home/ziyang10/openpi/.venv/lib/python3.11/site-packages/faiss`

### Step 11: Validate After Equivalent Install

#### Validation A: Import with `LD_LIBRARY_PATH`

Passed:

- `import faiss`
- GPU smoke test

#### Validation B: Import without `LD_LIBRARY_PATH`

Passed:

- `import faiss`

#### Validation C: GPU Smoke Test without `LD_LIBRARY_PATH`

Passed:

- `StandardGpuResources()`
- CPU-to-GPU index transfer
- add/search on GPU

### Final Practical Result

On this machine, after the equivalent install into `site-packages`, Faiss behaves like a persistent package for the `openpi/.venv` environment:

- no `PYTHONPATH` required
- no `LD_LIBRARY_PATH` required
- no recompilation required after reboot

This result is specific to the current machine and environment layout. Future environment changes can still affect this behavior.

## Troubleshooting and Likely Causes of Future Faiss Problems

If Faiss-related issues appear later, they will usually come from one of the following classes of problems.

### 1. `import faiss` Fails

Possible causes:

- `faiss` package directory missing from `site-packages`
- package copied incompletely
- Python is not the expected interpreter
- `sys.path` changed unexpectedly

Checks:

```bash
/home/ziyang10/openpi/.venv/bin/python -c "import sys, faiss; print(sys.executable); print(faiss.__file__)"
```

Solutions:

- verify that `faiss` exists under `openpi/.venv/lib/python3.11/site-packages`
- ensure you are using `/home/ziyang10/openpi/.venv/bin/python`
- if needed, recopy the built package from `build/lib/faiss`

### 2. Import Works but GPU APIs Fail

Possible causes:

- dynamic libraries cannot be resolved at runtime
- CUDA runtime libraries differ from what the built package expects
- the active shell or process has a different library environment than during validation

Checks:

```bash
/home/ziyang10/openpi/.venv/bin/python - <<'PY'
import faiss
print(hasattr(faiss, "StandardGpuResources"))
PY
```

Solutions:

- retry with:

```bash
export LD_LIBRARY_PATH=/home/ziyang10/faiss-build/install/lib:/home/ziyang10/.conda/envs/faiss-build/lib:/home/ziyang10/.conda/envs/faiss-build/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

- then rerun a minimal GPU smoke test
- if this fixes the issue, the problem is runtime library resolution

### 3. GPU Smoke Test Fails After a Future Environment Change

Possible causes:

- CUDA toolkit path changed
- Conda environment moved or was deleted
- Python environment was recreated
- system driver changed
- `openpi/.venv` was recreated and lost the copied `faiss` package

Checks:

- verify that `/home/ziyang10/.conda/envs/faiss-build` still exists
- verify that `/home/ziyang10/faiss-build/install/lib` still exists
- verify that `/home/ziyang10/openpi/.venv/lib/python3.11/site-packages/faiss` still exists
- confirm GPU visibility with PyTorch

Solutions:

- if the copied package disappeared, recopy it
- if the runtime libraries disappeared, restore them or rebuild Faiss
- if CUDA or the Python runtime changed substantially, rerun configure/build/install

### 4. Header or Build Binding Problems During Rebuild

Possible causes:

- CMake finds the wrong Python interpreter
- CMake finds the wrong NumPy include path
- Conda toolchain environment is not activated

Checks:

- inspect `CMakeCache.txt`
- confirm `Python_EXECUTABLE`
- confirm `Python_INCLUDE_DIR`
- confirm `CUDAToolkit_ROOT`

Solutions:

- rerun the explicit CMake configure command from the executed build record above
- do not rely on autodetection for Python if the environment changes

### 5. Warnings During CUDA Compilation

Observed in the successful build:

- repeated `warning #611-D` from NVCC

Interpretation:

- these warnings were benign in the successful build
- they are not a problem by themselves unless they are promoted to errors or followed by link/runtime failures

Solution:

- do not treat these warnings alone as a build failure
- only investigate if the build stops or the produced binaries misbehave

### 6. Shell Formatting and Environment Variable Errors

Possible causes:

- a long `export LD_LIBRARY_PATH=...` line is accidentally split by the shell or terminal paste

Symptoms:

- shell errors like `No such file or directory`

Solution:

- keep the export on one logical line
- or put it in a script file instead of pasting it manually

### 7. Recommended Recovery Order

If Faiss breaks in the future, use this order:

1. verify `openpi/.venv/bin/python`
2. verify `import faiss`
3. verify `StandardGpuResources`
4. retry with the explicit `LD_LIBRARY_PATH`
5. verify the copied package still exists in `site-packages`
6. verify the installed C++ libraries still exist under `/home/ziyang10/faiss-build/install/lib`
7. if still broken, rerun configure/build/install from the recorded commands

## References

- Faiss installation guide:
  - https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
- NVIDIA CUDA 12.6 Linux installation guide:
  - https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html
- OpenPI runtime and dependency definition:
  - [pyproject.toml](/home/ziyang10/openpi/pyproject.toml)
- OpenPI cache architecture:
  - [cache_system_architecture.md](/home/ziyang10/openpi/docs/cache_system_architecture.md)
