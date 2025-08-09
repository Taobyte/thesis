#!/bin/bash 

# setup environment 
conda activate thesis

# ressource specs
TIME=24:00:00

BLNNAME="bl_none"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia,wildppg,ieee lbw=a,b,c,d,e pw=a model=linear,hlinear,kalmanfilter,xgboost,hxgboost normalization=none use_wandb=True tune=False experiment=endo_only folds=fold_0,fold_1,fold_2"
sbatch -o "$BLNNAME" --time="$TIME" --wrap="$JOB"

BLGNAME="bl_global"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia,wildppg,ieee lbw=a,b,c,d,e pw=a model=gp normalization=global use_wandb=True tune=False experiment=endo_only folds=fold_0,fold_1,fold_2"
sbatch -o "$BLGNAME" --time="$TIME" --wrap="$JOB"

DLNAME="dl_global"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia,wildppg,ieee lbw=a,b,c,d,e pw=a model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts normalization=global use_wandb=True tune=False experiment=endo_only folds=fold_0,fold_1,fold_2"
sbatch -o "$DLNAME" --time="$TIME" --wrap="$JOB"
