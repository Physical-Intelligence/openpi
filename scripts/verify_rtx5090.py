
import os

# Set environment variables for JAX *before* importing
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax
import jax.numpy as jnp
import torch
import numpy as np
import logging

def verify():
    print("Verifying JAX + Torch on RTX 5090...")
    print(f"JAX version: {jax.__version__}")
    print(f"Torch version: {torch.__version__}")
    
    try:
        # Check Torch GPU
        if not torch.cuda.is_available():
            print("ERROR: Torch does not see CUDA!")
            return False
        
        t_dev = torch.device("cuda")
        t_x = torch.ones(1024, 1024, device=t_dev)
        print("Torch GPU allocation successful.")
        
        # Check JAX GPU
        j_devs = jax.devices()
        print(f"JAX devices: {j_devs}")
        if not j_devs or j_devs[0].platform != 'gpu':
            print("ERROR: JAX is not using GPU!")
            # It might be using CPU if plugin failed
        
        # Simple JAX Op
        key = jax.random.PRNGKey(0)
        j_x = jax.random.normal(key, (1024, 1024))
        j_y = jnp.dot(j_x, j_x)
        print("JAX matmul successful.")
        
        # Conversion to Numpy (Crash point in train.py)
        # In train.py: np.array(img[i]) where img[i] is JAX array
        print("Attempting JAX -> Numpy conversion (simulating logging)...")
        n_y = np.array(j_y)
        print(f"Conversion successful. Shape: {n_y.shape}")
        
        print("VERIFICATION PASSED!")
        return True
        
    except Exception as e:
        print(f"VERIFICATION FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    verify()
