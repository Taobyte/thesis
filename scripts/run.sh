#!/bin/bash 
#SBATCH -n 1 
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=8:00:00
#SBATCH --gpus=1

conda activate thesis
nvidia-smi 
python main.py path=cluster
