#!/bin/bash

# Load the module
module load python/anaconda

# Activate the Conda environment
#source /s/lovelace/h/nobackup/sangmi/hackathon/AAAI-2025/fire-asufm/env_fire/bin/activate

# Run the Python script
python main.py --seed 42 --dir_checkpoint ./checkpoints/  --epochs 500 --batch_size 32 