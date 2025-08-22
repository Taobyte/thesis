#!/bin/bash -l

#SBATCH --gpus=1
#SBATCH --gres=gpumem:24g
#SBATCH --mem-per-cpu=24G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --job-name=tune_job
#SBATCH --output=/cluster/project/holz/ckeusch/tune_logs/%x_%j.out

source ~/.bashrc
conda activate thesis

module load eth_proxy

nvidia-smi;python main.py --multirun hydra/sweeper=${1}_sweeper model=${1} normalization=${2} experiment=endo_exo n_trials=${3} lbw=${4} pw=${5} tune=True overfit=False
