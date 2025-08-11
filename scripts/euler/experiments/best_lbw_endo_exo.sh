#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

# TODO: TUNE DALIA & WILDPPG LOOKBACK WINDOWS!

# ressource specs
TIME=24:00:00

BLNNAME="best_endo_exo_bl_none"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=ieee lbw=lbw pw=a model=linear,hlinear,kalmanfilter,xgboost,hxgboost normalization=none use_wandb=True tune=False experiment=best_endo_exo folds=fold_0,fold_1,fold_2"
sbatch --job-name="$BLNNAME" -o "$BLNNAME" --time="$TIME" --wrap="$JOB"

BLGNAME="best_endo_exo_bl_global"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=ieee lbw=lbw pw=a model=gp,mlp normalization=global use_wandb=True tune=False experiment=best_endo_exo folds=fold_0,fold_1,fold_2"
sbatch --job-name="$BLGNAME" -o "$BLGNAME" --time="$TIME" --wrap="$JOB"

DLNAME="best_endo_exo_dl_global"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=ieee lbw=lbw pw=a model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts normalization=global use_wandb=True tune=False experiment=best_endo_exo folds=fold_0,fold_1,fold_2"
sbatch --job-name="$DLNAME" -o "$DLNAME" --time="$TIME" --wrap="$JOB"
