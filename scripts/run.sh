#!/bin/bash 


#SBATCH -n 1 
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00:00
#SBATCH --gpus=1

NUM_WORKERS="$1"

conda activate thesis
nvidia-smi 
python main.py path=cluster num_workers=${NUM_WORKERS}
