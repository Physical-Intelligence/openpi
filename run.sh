#!/bin/bash

# Generate timestamp in YYYYMMDDHHMMSS format
TIMESTAMP=$(date +"%Y%m%d%H%M%S")

# Training script for pi05 aloha simulation with automatic timestamp
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
python scripts/train.py pi05_aloha_sim_insertion_human --exp-name=epri_aloha_sim_insertion_pi05${TIMESTAMP}
# python scripts/train.py pi05_aloha_sim_transfer_cube_human --exp-name=epri_aloha_sim_insertion_pi05${TIMESTAMP} --resume
