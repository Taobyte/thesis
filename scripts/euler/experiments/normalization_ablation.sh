#!/bin/bash

source ~/.bashrc
conda activate thesis

TIME="24:00:00"
GLOBAL_NAME="global_norm_exp"

GLOBAL_JOB="python main.py --multirun \
    hydra/launcher=submitit_slurm \
    dataset=dalia,wildppg,ieee \
    lbw=d \
    pw=a \
    model=linear,mole,msar,kalmanfilter,xgboost,gp,mlp,timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx \
    use_wandb=True \
    tune=False \
    experiment=global_z_norm,local_z_norm,min_max_norm,no_norm,difference \
    folds=fold_0,fold_1,fold_2 \
    seed=0,1,2"

sbatch --job-name="$GLOBAL_NAME" \
       --output="$GLOBAL_NAME".out \
       --time="$TIME" \
       --wrap="$GLOBAL_JOB"

LOCAL_NAME="local_norm_exp"
LOCAL_JOB="python main.py --multirun \
    hydra/launcher=submitit_slurm \
    dataset=ldalia,lwildppg,lieee \
    lbw=d \
    pw=a \
    model=linear,mole,msar,kalmanfilter,xgboost,gp,mlp,timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx \
    use_wandb=True \
    tune=False \
    experiment=global_z_norm,local_z_norm,min_max_norm,no_norm,difference \
    seed=0,1,2"

sbatch --job-name="$LOCAL_NAME" \
       --output="$LOCAL_NAME".out \
       --time="$TIME" \
       --wrap="$LOCAL_JOB"