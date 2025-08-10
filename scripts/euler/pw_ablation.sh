#!/bin/bash 

# setup environment 
conda activate thesis

# ressource specs
TIME=24:00:00

BLNNAME="pw_endo_exo_bl_none"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia,wildppg,ieee lbw=b pw=a,b,c model=linear,hlinear,kalmanfilter,xgboost,hxgboost normalization=none use_wandb=True tune=False experiment=endo_exo folds=fold_0,fold_1,fold_2"
sbatch --job-name="$BLNNAME" -o "$BLNNAME" --time="$TIME" --wrap="$JOB"

BLGNAME="pw_endo_exo_bl_global"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia,wildppg,ieee lbw=b pw=a,b,c model=gp,mlp normalization=global use_wandb=True tune=False experiment=endo_exo folds=fold_0,fold_1,fold_2"
sbatch --job-name="$BLGNAME" -o "$BLGNAME" --time="$TIME" --wrap="$JOB"

DLNAME="pw_endo_exo_dl_global"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia,wildppg,ieee lbw=b pw=a,b,c model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts normalization=global use_wandb=True tune=False experiment=endo_exo folds=fold_0,fold_1,fold_2"
sbatch --job-name="$DLNAME" -o "$DLNAME" --time="$TIME" --wrap="$JOB"
