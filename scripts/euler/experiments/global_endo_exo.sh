#!/bin/bash

source ~/.bashrc
conda activate thesis

# resource specs
TIME=24:00:00

LBW=d  # d = 30
PW=a   # a = 3

# ------------------- DALIA -------------------
DALIA_RUNS="exo_dalia"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia lbw=$LBW pw=$PW model=linear,kalmanfilter,nbeatsx normalization=global use_norm_dl=False use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2 seed=0,1,2"
sbatch --export=ALL --job-name="${DALIA_RUNS}_global"     -o "${DALIA_RUNS}_global_%j.out"     --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia lbw=$LBW pw=$PW model=mole,gp,adamshyper,patchtst normalization=global use_norm_dl=True use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2 seed=0,1,2"
sbatch --export=ALL --job-name="${DALIA_RUNS}_local"      -o "${DALIA_RUNS}_local_%j.out"      --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia lbw=$LBW pw=$PW model=msar,xgboost,mlp,timesnet,simpletm,timexer,gpt4ts normalization=difference use_norm_dl=False use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2 seed=0,1,2"
sbatch --export=ALL --job-name="${DALIA_RUNS}_difference" -o "${DALIA_RUNS}_difference_%j.out" --time="$TIME" --wrap="$JOB"

# ------------------- WILDPPG -------------------
WILDPPG_RUNS="exo_wildppg"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=wildppg lbw=$LBW pw=$PW model=kalmanfilter normalization=none use_norm_dl=False use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2 seed=0,1,2"
sbatch --export=ALL --job-name="${WILDPPG_RUNS}_none"     -o "${WILDPPG_RUNS}_none_%j.out"     --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=wildppg lbw=$LBW pw=$PW model=gp,nbeatsx normalization=global use_norm_dl=False use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2 seed=0,1,2"
sbatch --export=ALL --job-name="${WILDPPG_RUNS}_global"     -o "${WILDPPG_RUNS}_global_%j.out"     --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=wildppg lbw=$LBW pw=$PW model=xgboost,timesnet,simpletm,patchtst,timexer normalization=global use_norm_dl=True use_norm_baseline=True use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2 seed=0,1,2"
sbatch --export=ALL --job-name="${WILDPPG_RUNS}_local"      -o "${WILDPPG_RUNS}_local_%j.out"      --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=wildppg lbw=$LBW pw=$PW model=linear,mole,msar,mlp,adamshyper,gpt4ts normalization=difference use_norm_dl=False use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2 seed=0,1,2"
sbatch --export=ALL --job-name="${WILDPPG_RUNS}_difference" -o "${WILDPPG_RUNS}_difference_%j.out" --time="$TIME" --wrap="$JOB"

# ------------------- IEEE -------------------
IEEE_RUNS="exo_ieee"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=ieee lbw=$LBW pw=$PW model=linear normalization=global use_norm_dl=False use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2 seed=0,1,2"
sbatch --export=ALL --job-name="${IEEE_RUNS}_global" -o "${IEEE_RUNS}_global_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=ieee lbw=$LBW pw=$PW model=mole,msar,kalmanfilter,xgboost,gp,mlp,timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx normalization=difference use_norm_dl=False use_wandb=True experiment=endo_exo,endo_only folds=fold_0,fold_1,fold_2 seed=0,1,2"
sbatch --export=ALL --job-name="${IEEE_RUNS}_difference" -o "${IEEE_RUNS}_difference_%j.out" --time="$TIME" --wrap="$JOB"
