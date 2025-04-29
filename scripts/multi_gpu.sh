#!/bin/bash -l

#SBATCH --gpus=4
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=24G
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=4

module load eth_proxy
python main.py model=timellm use_multi_gpu=True
