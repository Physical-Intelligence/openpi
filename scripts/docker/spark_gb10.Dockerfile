# syntax=docker/dockerfile:1.7

# OpenPI pi0.5 runtime for NVIDIA GB10 / DGX Spark.
#
# The upstream Dockerfile installs PyTorch 2.7 and CUDA-12 JAX wheels. Those
# wheels do not contain GB10 (sm_121) kernels. This image instead keeps the
# NVIDIA 26.01 PyTorch + CUDA 13.1 stack that is validated on the target Spark,
# while retaining CPU-only JAX for Orbax checkpoint conversion.
ARG BASE_IMAGE=codex/mjlab-bench:20260809-egl
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="OpenPI pi0.5 for NVIDIA GB10"
LABEL org.opencontainers.image.description="CUDA 13.1 PyTorch runtime with CPU JAX checkpoint conversion"

ENV DEBIAN_FRONTEND=noninteractive
ENV JAX_PLATFORMS=cpu
ENV OPENPI_DATA_HOME=/openpi_assets
ENV PYTHONUNBUFFERED=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV VIRTUAL_ENV=/opt/openpi-venv
ENV PATH=/opt/openpi-venv/bin:${PATH}

WORKDIR /opt/openpi

RUN apt-get update \
    && apt-get install -y --no-install-recommends git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv --system-site-packages "${VIRTUAL_ENV}"

# Export the locked dependency graph, but prune packages that would replace the
# NVIDIA-provided PyTorch/TorchVision pair or pull a parallel CUDA-12 runtime.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY packages/openpi-client/pyproject.toml packages/openpi-client/pyproject.toml
COPY packages/openpi-client/src packages/openpi-client/src
RUN uv export --quiet \
        --frozen \
        --no-dev \
        --no-emit-project \
        --no-emit-workspace \
        --prune torch \
        --prune torchvision \
        --prune jax-cuda12-plugin \
        --prune jax-cuda12-pjrt \
        --prune gym-aloha \
        --prune dm-control \
        --prune labmaze \
        --prune mujoco \
        --output-file /tmp/spark-requirements.txt \
    && ! grep -Eq '^(torch|torchvision|jax-cuda12|nvidia-.*-cu12)([= @;]|$)' /tmp/spark-requirements.txt \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-deps --requirement /tmp/spark-requirements.txt \
    && rm /tmp/spark-requirements.txt

COPY src src
COPY scripts scripts
COPY examples examples
COPY packages packages

RUN uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-deps --editable packages/openpi-client --editable . \
    && python3 -c "import pathlib, shutil, transformers; target = pathlib.Path(transformers.__file__).parent; [shutil.copy2(path, target / path.name) for path in pathlib.Path('src/openpi/models_pytorch/transformers_replace').glob('*') if path.is_file()]; [shutil.copytree(path, target / path.name, dirs_exist_ok=True) for path in pathlib.Path('src/openpi/models_pytorch/transformers_replace').glob('*') if path.is_dir()]" \
    && JAX_PLATFORMS=cpu python3 -c "import jax, torch, transformers; assert jax.default_backend() == 'cpu'; assert torch.__version__.startswith('2.10.0a0'); assert transformers.__version__ == '4.53.2'; print('OpenPI GB10 dependency contract verified')"

CMD ["bash"]
