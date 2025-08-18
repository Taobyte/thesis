#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

# ressource specs
TIME=24:00:00

BLNNAME="bl_difference"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia,wildppg,ieee lbw=lbw pw=a model=linear,hlinear,kalmanfilter,xgboost,hxgboost,gp,mlp normalization=difference use_wandb=True tune=False experiment=difference folds=fold_0,fold_1,fold_2"
sbatch --job-name="$BLNNAME" -o "$BLNNAME" --time="$TIME" --wrap="$JOB"

