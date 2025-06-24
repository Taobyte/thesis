#!/bin/bash -l

#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=24G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --job-name=tune_job
#SBATCH --output=/cluster/project/holz/ckeusch/tune_logs/%x_%j.out

source ~/.bashrc
conda activate thesis

module load eth_proxy

nvidia-smi;python main.py --multirun hydra/sweeper=${1}_sweeper model=${1} normalization=${2} use_dynamic_features=${3} look_back_window=${4} prediction_window=${5} tune=True overfit=False num_workers=8
