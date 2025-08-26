#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

# ressource specs
TIME=24:00:00

DATASET=dalia
EXPERIMENT=endo_exo
LBW=d
PW=a

DALIA_RUNS="eff_dalia"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=$DATASET lbw=$LBW pw=$PW model=linear,kalmanfilter,gp,adamshyper,nbeatsx normalization=global use_wandb=True tune=False experiment=efficiency use_efficiency_callback=True folds=fold_0,fold_1,fold_2"
sbatch --job-name="$DALIA_RUNS_global" -o "$DALIA_RUNS_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=$DATASET lbw=$LBW pw=$PW model=timesnet,patchtst,timexer,gpt4ts normalization=global use_norm_dl=True use_wandb=True tune=False experiment=efficiency use_efficiency_callback=True folds=fold_0,fold_1,fold_2"
sbatch --job-name="$DALIA_RUNS_local" -o "$DALIA_RUNS_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=$DATASET lbw=$LBW pw=$PW model=mole,msar,xgboost,mlp,simpletm normalization=difference use_wandb=True tune=False experiment=efficiency use_efficiency_callback=True folds=fold_0,fold_1,fold_2"
sbatch --job-name="$DALIA_RUNS_difference" -o "$DALIA_RUNS_%j.out" --time="$TIME" --wrap="$JOB"


