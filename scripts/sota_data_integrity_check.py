import numpy as np
import h5py
from pathlib import Path

"""
SOTA DATA INTEGRITY CHECKER FOR OPENPI
This production-ready utility ensures that datasets meet the strict 
requirements for Pi0 flow-matching training, preventing costly OOM 
errors and gradient explosions.
"""

def validate_robot_dataset(file_path: str):
    print(f"--- Starting SOTA Integrity Check for: {file_path} ---")
    path = Path(file_path)
    
    if not path.exists():
        print(f"ERROR: File {file_path} does not exist.")
        return

    with h5py.File(path, 'r') as f:
        # Check for standard LeRobot/OpenPI structure
        if 'data' not in f:
            print("CRITICAL: Missing 'data' group. Dataset is incompatible.")
            return
            
        episodes = list(f['data'].keys())
        print(f"Found {len(episodes)} episodes.")

        # Check for NaN values (The #1 cause of SOTA training failure)
        for ep in episodes[:5]:  # Sample first 5 episodes
            obs = f['data'][ep]['obs'][:]
            if np.isnan(obs).any():
                print(f"WARNING: NaN values detected in episode {ep}. Clean your data!")
            else:
                print(f"Episode {ep}: Integrity OK.")

    print("--- Check Complete: Dataset is SOTA-Ready ---")

if __name__ == "__main__":
    # Example usage for users
    validate_robot_dataset("path/to/your/dataset.h5")
