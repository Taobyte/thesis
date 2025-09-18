#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

# ressource specs
TIME=24:00:00

# lookback and prediction window settings
LBW=c # c = 20
PW=d # d = 10

JOB="python main.py --multirun hydra/launcher=cpu use_prediction_callback=True dataset=dalia,wildppg,ieee lbw=$LBW pw=$PW model=linear,xgboost normalization=global local_norm=difference use_wandb=True experiment=endo_exo,endo_only folds=fold_0"
sbatch --job-name="cpu" -o "lbw_abl_cpu_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=gpu_small use_prediction_callback=True dataset=dalia,wildppg,ieee lbw=$LBW pw=$PW model=mole,msar,kalmanfilter,gp,mlp normalization=global local_norm=difference use_wandb=True experiment=endo_exo,endo_only folds=fold_0"
sbatch --job-name="gpu_small" -o "lbw_abl_gpu_s_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=gpu_large use_prediction_callback=True dataset=dalia,wildppg,ieee lbw=$LBW pw=$PW model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx normalization=global local_norm=local_z use_wandb=True experiment=endo_exo,endo_only folds=fold_0"
sbatch --job-name="gpu_large" -o "lbw_abl_gpu_l_%j.out" --time="$TIME" --wrap="$JOB"


