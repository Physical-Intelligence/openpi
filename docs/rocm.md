# Training on AMD GPUs (ROCm)

The default setup targets NVIDIA GPUs: `pyproject.toml` pins the CUDA builds of `torch` and `jax`, so `uv sync` cannot be used on a ROCm host. Instead, `scripts/docker/train_rocm.Dockerfile` builds on top of a ROCm base image that already ships a matching PyTorch, and installs the remaining openpi dependencies without disturbing it.
This covers PyTorch training only. Everything under [PyTorch Support](../README.md#pytorch-support) in the main README applies, with the same feature limitations. JAX training is not supported here. Tested on 8×MI308X (gfx942) with ROCm 7.14 and PyTorch 2.12. Other gfx942 parts (MI300X, MI325X) use the same base image and should work; older generations have not been tested.

## Requirements
 
- A ROCm-capable GPU and the `amdgpu` kernel driver on the host. Check with `rocm-smi`.
- Docker. Unlike NVIDIA, there is no container toolkit to install — GPU access comes from passing `/dev/kfd` and `/dev/dri` into the container and adding the `video` and `render` groups, which `compose_rocm.yml` already does.

## Quick start

```bash
docker compose -f scripts/docker/compose_rocm.yml up -d --build
docker compose -f scripts/docker/compose_rocm.yml exec openpi_rocm bash
```

The repo is mounted at `/app`, so local edits take effect without rebuilding. Checkpoints and datasets are large, so point `OPENPI_DATA_HOME` at a local disk rather than a network share — on NFS the data loader becomes the bottleneck:

```bash
OPENPI_DATA_HOME=/mnt/local/openpi docker compose -f scripts/docker/compose_rocm.yml up -d
```

Verify that the GPUs are visible from inside the container:

```bash
python3 -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

## Running training

The workflow is the same as [Finetuning with PyTorch](../README.md#finetuning-with-pytorch), except that commands run directly rather than through `uv run`. Convert a base checkpoint, compute norm stats, then train:

```bash
python3 examples/convert_jax_model_to_pytorch.py \
    --config_name <config name> \
    --checkpoint_dir /path/to/jax/base/model \
    --output_path /path/to/pytorch/base/model

python3 scripts/compute_norm_stats.py --config-name <config name>

# single-node training
torchrun --standalone --nnodes=1 --nproc_per_node=<gpus_per_node> scripts/train_pytorch.py <config_name> --exp_name <run_name> --pytorch-weight-path /path/to/pytorch/base/model --num-workers 8

# multi-node training
torchrun --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval> --pytorch-weight-path /path/to/pytorch/base/model
```

## How the image differs from the uv setup

Three deviations from `pyproject.toml` are needed to keep the ROCm stack intact. They are applied in the Dockerfile and explained here because they are easy to undo by accident.

**`uv sync` is not used.** Installing openpi's pinned `torch==2.7.1` and `jax[cuda12]` would replace the ROCm builds with CUDA wheels. The Dockerfile instead generates a constraints file from the versions already present in the base image and passes it to every `pip install`. Packages that would otherwise pull in torch — openpi, openpi-client and lerobot — are installed with `--no-deps`.

**numpy stays on 2.x** rather than the `<2.0.0` that `pyproject.toml` requires. The ROCm extensions in the base image are compiled against the numpy 2 ABI and break if numpy is downgraded. jax 0.5.3, flax, orbax and the openpi data pipeline all work under numpy 2.

**JAX is installed as the CPU build** and `JAX_PLATFORMS=cpu` is set. openpi imports jax for its configs, data transforms and orbax checkpoint loading, but all GPU work goes through PyTorch, so jax should not claim GPU memory.

As a result `pip` prints dependency conflict warnings during the build, and `pip check` reports the numpy and torch pins above alongside a handful of unrelated packages that the base image ships and openpi never imports. These are expected. The final build layer imports the model module and instantiates a training config, so the build still fails if the environment is genuinely broken.

## Known issues

**`--overwrite` is racy under multi-GPU training.** `scripts/train_pytorch.py` calls `shutil.rmtree(config.checkpoint_dir)` without guarding on the main process, so every rank deletes the same directory concurrently and they remove files out from under each other. This surfaces as `FileNotFoundError` on `norm_stats.json` or on the checkpoint directory itself. It is not specific to ROCm. Use a fresh `--exp_name` for each run instead of `--overwrite`.
