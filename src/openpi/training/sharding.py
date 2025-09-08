import contextlib
import logging

import jax
import numpy as np

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
# In FSDP, we shard the data across both the batch and FSDP axes.
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


def make_mesh(num_fsdp_devices: int) -> jax.sharding.Mesh:
    if jax.device_count() % num_fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {jax.device_count()} must be divisible by the number of FSDP devices {num_fsdp_devices}."
        )
    mesh_shape = (jax.device_count() // num_fsdp_devices, num_fsdp_devices)
    return jax.make_mesh(mesh_shape, (BATCH_AXIS, FSDP_AXIS))


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """Plumbing the mesh deep into the module tree is extremeley cumbersome; until the JAX team lands a better API, a
    custom context manager like this one is the recommended way to maintain a reference to a global mesh. This is only used
    in `activation_sharding_constraint` below."""
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None


def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS))
    )


def replicate_sharding_constraint(pytree):
    """Replicate the tensor across the active mesh (explicit all-gather).

    If there is no active mesh, this is a no-op.
    """
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec())
    )


def megatron_mlp_input_constraint(pytree):
    """Apply Megatron tensor parallel input sharding: P("batch", None, None).
    
    For MLP blocks in tensor parallel, input should be batch-sharded but FSDP-replicated.
    If there is no active mesh, this is a no-op.
    """
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec("batch", None, None))
    )


def megatron_mlp_output_constraint(pytree):
    """Apply Megatron tensor parallel output sharding: P("batch", None, "fsdp").
    
    For MLP blocks in tensor parallel, output should be batch-sharded and FSDP-sharded.
    If there is no active mesh, this is a no-op.
    """
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree, jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec("batch", None, "fsdp"))
    )


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 4 MiB
    log: bool = False,
):
    """Apply FSDP sharding to a pytree of arrays based on the mesh shape.

    Args:
        pytree: A pytree to be apply sharding specified by the mesh, note that only array types (eg. contains .shape attr)
          will be considered for sharding.
        mesh: The mesh being used for applying sharding on to pytree.
        min_size_mbytes: The minimum size of the array in MiB to be considered for sharding, any array smaller than this
          will be replicated.
        log: If true, will log the sharding decisions for arrays that are being considered for sharding.

    Returns:
        The sharded pytree.
    """
    min_size_bytes = min_size_mbytes * 2**20

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        # if fsdp is not actually going to be used, replicate everything to avoid extraneous logging
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate scalar and vector arrays
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate small arrays
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # shard matrices and larger tensors along the largest axis that is divisible by the fsdp dimension
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(
                        f"Sharding {jax.tree_util.keystr(kp)} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}"
                    )
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

        # replicate if no valid sharding was found
        if log:
            logging.warning(
                f"Could not find a valid sharding for {jax.tree_util.keystr(kp)} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)


def megatron_tensor_parallel_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    column_parallel_names: list[str],
    row_parallel_names: list[str],
    shard_axis: str = 'fsdp',
    log: bool = False,
):
    """Apply Megatron-style tensor parallel sharding based on parameter name patterns.
    
    Args:
        pytree: A pytree to apply sharding to.
        mesh: The mesh being used for sharding.
        column_parallel_names: List of parameter names that should be column-parallel sharded.
                              If None, uses default MLP patterns.
        row_parallel_names: List of parameter names that should be row-parallel sharded.
                           If None, uses default MLP patterns.
        shard_axis: The mesh axis to shard along (default: 'fsdp').
        log: If true, will log the sharding decisions.
    
    Returns:
        The sharded pytree.
    """
    # Validate inputs: names must be provided explicitly
    if not isinstance(column_parallel_names, list) or len(column_parallel_names) == 0:
        raise ValueError("column_parallel_names must be a non-empty list of name patterns")
    if not isinstance(row_parallel_names, list) or len(row_parallel_names) == 0:
        raise ValueError("row_parallel_names must be a non-empty list of name patterns")
    
    def _shard_arr(kp, array):
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        
        key_path = jax.tree_util.keystr(kp)
        shape = array.shape
        
        # Check for column parallel patterns
        for pattern in column_parallel_names:
            if pattern in key_path:
                # Column parallel: shard last dimension
                assert len(shape) >= 2, (
                    f"Column-parallel tensor must have rank >= 2 at {key_path}, got shape {shape}"
                )
                assert shape[-1] % mesh.shape[shard_axis] == 0, (
                    f"Last dimension {shape[-1]} not divisible by mesh {shard_axis} size {mesh.shape[shard_axis]} "
                    f"for {key_path}"
                )
                spec = [None] * len(shape)
                spec[-1] = shard_axis
                if log:
                    logging.info(f"Column parallel sharding {key_path}: {spec}")
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
        
        # Check for row parallel patterns  
        for pattern in row_parallel_names:
            if pattern in key_path:
                # Row parallel: shard first dimension
                assert len(shape) >= 2, (
                    f"Row-parallel tensor must have rank >= 2 at {key_path}, got shape {shape}"
                )
                assert shape[0] % mesh.shape[shard_axis] == 0, (
                    f"First dimension {shape[0]} not divisible by mesh {shard_axis} size {mesh.shape[shard_axis]} "
                    f"for {key_path}"
                )
                spec = [None] * len(shape)
                spec[0] = shard_axis
                if log:
                    logging.info(f"Row parallel sharding {key_path}: {spec}")
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
        
        # Default: use FSDP sharding for remaining parameters
        return fsdp_sharding({key_path: array}, mesh, log=log)[key_path]
    
    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)
