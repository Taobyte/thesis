#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

# ressource specs
TIME=24:00:00
LBW=d # d = 30 

NAME="local_runs"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=ldalia,lwildppg,lieee lbw=lbw pw=a model=linear,mole,msar,kalmanfilter,gp,xgboost,mlp,timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx normalization=difference use_wandb=True tune=False experiment=endo_exo folds=fold_0,fold_1,fold_2"
sbatch --job-name="$NAME" -o "$NAME_%j.out" --time="$TIME" --wrap="$JOB"
