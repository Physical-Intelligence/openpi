# Dockerfile for training openpi on AMD GPUs (ROCm).
#
# The base image ships ROCm 7.14 and a matching PyTorch 2.12 build for gfx942
# (MI300 series). openpi's pyproject pins the CUDA builds of torch and jax, so
# this image does not use `uv sync`; it installs the dependencies under a
# constraints file that keeps the pre-installed ROCm stack in place.
#
# Build the container:
# docker build . -t openpi_rocm -f scripts/docker/train_rocm.Dockerfile

# Run the container:
# docker compose -f scripts/docker/compose_rocm.yml up -d

# See docs/rocm.md for hardware requirements and known issues.

FROM rocm/primus:v26.4@sha256:8c8ecc6fe14b5061423cc82517831e3515f6eb15647ee6d746fd6a0c1963da24

# Kept in sync with pyproject.toml.
ARG LEROBOT_REF=0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
ARG TRANSFORMERS_VER=4.53.2

ENV PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GIT_LFS_SKIP_SMUDGE=1 \
    ROCPROFILER_LOG_LEVEL=fatal

WORKDIR /app

# Pin the ROCm stack. pyproject asks for torch==2.7.1 (CUDA), jax[cuda12] and
# numpy<2, all of which would replace the ROCm builds that the base image ships.
# Generate a constraints file from the versions already installed and pass it to
# every install below. numpy stays on 2.x: downgrading breaks the ROCm
# extensions, which are compiled against the numpy 2 ABI.
RUN set -eux; \
    python3 -c "import importlib.metadata as m; \
names='torch torchvision triton numpy pillow tokenizers huggingface_hub safetensors av'.split(); \
open('/etc/openpi-constraints.txt','w').write(''.join(n+'=='+m.version(n)+chr(10) for n in names))"; \
    cat /etc/openpi-constraints.txt

# JAX on CPU. openpi's configs, data transforms and orbax checkpoint loading
# import jax, but all GPU work goes through PyTorch.
RUN pip install -c /etc/openpi-constraints.txt \
    "jax==0.5.3" "jaxlib==0.5.3" "flax==0.10.2" "orbax-checkpoint==0.11.13" \
    "ml-dtypes==0.4.1" "tensorstore==0.1.74" "chex==0.1.90" "beartype==0.19.0" \
    "jaxtyping==0.2.36" "equinox>=0.11.8" "augmax>=0.3.4" "dm-tree>=0.1.8" \
    "flatbuffers>=24.3.25" "ml_collections==1.0.0" "treescope>=0.1.7"

# Remaining runtime dependencies.
RUN pip install -c /etc/openpi-constraints.txt \
    "transformers==${TRANSFORMERS_VER}" \
    "imageio[ffmpeg]>=2.36.1" "numpydantic>=1.6.6" "opencv-python-headless>=4.10.0.84" \
    "polars>=1.30.0" "rich>=14.0.0" "tqdm-loggable>=0.2" "typing-extensions>=4.12.2" \
    "tyro>=0.9.5" "wandb>=0.19.1" "filelock>=3.16.1" "sentencepiece>=0.2.0" \
    "fsspec[gcs]>=2024.6.0" \
    jsonlines deepdiff draccus termcolor

# LeRobot, for its datasets module. --no-deps because it pulls CUDA torch; its
# transitive dependencies are covered above.
RUN pip install --no-deps -c /etc/openpi-constraints.txt \
    "lerobot @ git+https://github.com/huggingface/lerobot@${LEROBOT_REF}"

# Copied after the dependencies so that editing the repo does not invalidate the
# layers above.
COPY . /app

# Editable installs so that mounting the repo over /app at runtime picks up
# local edits. Same transformers patch as the uv setup in the main README:
# it adds AdaRMS, activation precision control and a read-only KV cache, without
# which PI0Pytorch fails to initialize.
RUN set -eux; \
    pip install --no-deps -e ./packages/openpi-client -e .; \
    cp -r src/openpi/models_pytorch/transformers_replace/* \
        "$(python3 -c 'import os, transformers; print(os.path.dirname(transformers.__file__))')/"; \
    python3 -c "from transformers.models.siglip import check; \
assert check.check_whether_transformers_replace_is_installed_correctly()"; \
    JAX_PLATFORMS=cpu python3 -c "\
import jax, openpi.training.config as c, openpi.models_pytorch.pi0_pytorch, torch; \
print('torch', torch.__version__, '| jax', jax.__version__, jax.devices()); \
print('config ok:', c.get_config('pi05_libero').name)"

# /openpi_assets is where compose_rocm.yml mounts OPENPI_DATA_HOME.
ENV OPENPI_DATA_HOME=/openpi_assets \
    HF_HOME=/openpi_assets/hf \
    JAX_PLATFORMS=cpu

CMD ["/bin/bash"]
