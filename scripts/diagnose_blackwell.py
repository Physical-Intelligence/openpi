
import logging
import jax
import jax.numpy as jnp
import numpy as np
import openpi.training.sharding as sharding

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"JAX version: {jax.__version__}")
    logging.info(f"Devices: {jax.devices()}")

    # Simulate the environment
    mesh = sharding.make_mesh(num_fsdp_devices=1)
    
    # Create a dummy batch of images (sharded)
    # Batch size 32, 224x224x3
    B, H, W, C = 32, 224, 224, 3
    # Sharding spec from train.py: DATA_AXIS = ('batch', 'fsdp')
    # If FSDP=1, then axis 1 is size 1.
    # Mesh shape (N_devices, 1).
    # Sharding spec for data: PartitionSpec('batch', 'fsdp')
    # So split along batch dimension.
    
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    
    key = jax.random.key(0)
    # Create array on device with sharding
    logging.info("Creating sharded array...")
    images = jax.random.uniform(key, (B, H, W, C), dtype=jnp.float32)
    images = jax.device_put(images, data_sharding)
    jax.block_until_ready(images)
    logging.info("Sharded array created.")

    # Reproduce the crash
    logging.info("Attempting to access single slice: np.array(images[0])...")
    try:
        # This corresponds to `np.array(img[i])` in the loop
        slice_0 = images[0]
        # Copy slice to host
        host_slice = np.array(slice_0)
        logging.info("Success: np.array(images[0])")
    except Exception as e:
        logging.error(f"Failed: np.array(images[0]) with error: {e}")
        # If this fails, Try workaround
        logging.info("Attempting workaround: np.array(images)[0]...")
        try:
            full_host = np.array(images)
            host_slice_workaround = full_host[0]
            logging.info("Success: np.array(images)[0]")
        except Exception as e2:
            logging.error(f"Failed workaround: np.array(images)[0] with error: {e2}")

if __name__ == "__main__":
    main()
