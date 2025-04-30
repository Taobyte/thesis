#!/bin/bash -l

#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=24G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=tune_job
#SBATCH --output=/cluster/project/holz/ckeusch/tune_logs/%x_%j.out

source ~/.bashrc
conda activate thesissource ./bashrc

module load eth_proxy

python main.py --multirun hydra/sweeper=${1}_sweeper model=${1} tune=True overfit=False num_workers=8
