#!/bin/bash

source ~/.bashrc
conda activate thesis

TIME="24:00:00"
BLNNAME="norm_experiments"

JOB="python main.py --multirun \
    hydra/launcher=submitit_slurm \
    dataset=dalia,wildppg,ieee,ldalia,lwildppg,lieee \
    lbw=d \
    pw=a \
    model=linear,mole,msar,kalmanfilter,xgboost,gp,mlp,timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx \
    use_wandb=True \
    tune=False \
    experiment=global_z_norm,local_z_norm,min_max_norm,no_norm,difference \
    folds=fold_0,fold_1,fold_2 \
    seed=0,1,2"

sbatch --job-name="$BLNNAME" \
       --output="$BLNNAME".out \
       --time="$TIME" \
       --wrap="$JOB"
