#!/bin/bash -l

#SBATCH --tasks-per-node=4
#SBATCH --gpus=rtx_3090:4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=24g
#SBATCH --nodes=1
#SBATCH --time=0-02:00:00

# setup
module load eth_proxy
conda activate thesis

srun python main.py model=timellm use_multi_gpu=True overfit=False use_wandb=True look_back_window=$1 prediction_window=$2 use_dynamic_features=$3 experiment=$4

