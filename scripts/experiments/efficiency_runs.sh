#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

# ressource specs
TIME=24:00:00

DATASET=dalia
LBW=d
PW=a

DALIA_RUNS="eff_dalia"
JOB="python main.py --multirun hydra/launcher=slurm_eff dataset=$DATASET lbw=$LBW pw=$PW model=linear,mole,msar,kalmanfilter,xgboost,gp,mlp normalization=difference use_wandb=True experiment=efficiency use_efficiency_callback=True folds=fold_0,fold_1,fold_2 seed=0,1,2"
sbatch --job-name="$DALIA_RUNS_global" -o "$DALIA_RUNS_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=slurm_eff dataset=$DATASET lbw=$LBW pw=$PW model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx normalization="global" use_wandb=True experiment=efficiency use_efficiency_callback=True folds=fold_0,fold_1,fold_2 seed=0,1,2"
sbatch --job-name="$DALIA_RUNS_difference" -o "$DALIA_RUNS_%j.out" --time="$TIME" --wrap="$JOB"


