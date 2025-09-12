#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

# ressource specs
TIME=24:00:00



JOB="python main.py --multirun hydra/launcher=cpu dataset=dalia,wildppg,ieee lbw=d pw=b,c,d,e model=linear,xgboost normalization=global local_norm=difference use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2"
sbatch --job-name="cpu" -o "lbw_abl_cpu_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=gpu_small dataset=dalia,wildppg,ieee lbw=d pw=b,c,d,e model=mole,msar,kalmanfilter,gp,mlp normalization=global local_norm=difference use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2"
sbatch --job-name="gpu_small" -o "lbw_abl_gpu_s_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=gpu_large dataset=dalia,wildppg,ieee lbw=d pw=b,c,d,e model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx normalization=global local_norm=local_z use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2"
sbatch --job-name="gpu_large" -o "lbw_abl_gpu_l_%j.out" --time="$TIME" --wrap="$JOB"


