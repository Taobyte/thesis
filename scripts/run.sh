#!/bin/bash 


#SBATCH -n 1 
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
#SBATCH --gpus=1

module load eth_proxy # used for logging to wandb

NUM_WORKERS="$1"

nvidia-smi 
python main.py path=cluster num_workers=${NUM_WORKERS}
