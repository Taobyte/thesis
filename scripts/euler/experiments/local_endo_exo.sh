#!/bin/bash

source ~/.bashrc
conda activate thesis

# resource specs
TIME=24:00:00

LBW=d  # d = 30
PW=a   # a = 3

# ------------------- DALIA -------------------
DALIA_RUNS="exo_ldalia"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=ldalia lbw=$LBW pw=$PW model=linear normalization=global local_norm=lnone use_wandb=True experiment=endo_exo,endo_only seed=0"
sbatch --export=ALL --job-name="${DALIA_RUNS}_global"     -o "${DALIA_RUNS}_global_%j.out"     --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=ldalia lbw=$LBW pw=$PW model=mole,msar,kalmanfilter,xgboost,gp,mlp,timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx normalization=global local_norm=difference use_wandb=True experiment=endo_exo,endo_only seed=0"
sbatch --export=ALL --job-name="${DALIA_RUNS}_difference" -o "${DALIA_RUNS}_difference_%j.out" --time="$TIME" --wrap="$JOB"

# ------------------- WILDPPG -------------------
WILDPPG_RUNS="exo_lwildppg"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=lwildppg lbw=$LBW pw=$PW model=gp normalization=minmax local_norm=lnone use_wandb=True experiment=endo_exo,endo_only seed=0"
sbatch --export=ALL --job-name="${WILDPPG_RUNS}_minmax"      -o "${WILDPPG_RUNS}_minmax_%j.out"      --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=lwildppg lbw=$LBW pw=$PW model=mole,xgboost,mlp,timesnet,simpletm,adamshyper,patchtst,timexer,nbeatsx normalization=global local_norm=local_z use_wandb=True experiment=endo_exo,endo_only seed=0"
sbatch --export=ALL --job-name="${WILDPPG_RUNS}_local"      -o "${WILDPPG_RUNS}_local_%j.out"      --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=lwildppg lbw=$LBW pw=$PW model=linear,msar,kalmanfilter,gpt4ts normalization=global local_z=difference use_wandb=True experiment=endo_exo,endo_only seed=0"
sbatch --export=ALL --job-name="${WILDPPG_RUNS}_difference" -o "${WILDPPG_RUNS}_difference_%j.out" --time="$TIME" --wrap="$JOB"

# ------------------- IEEE -------------------
IEEE_RUNS="exo_lieee"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=lieee lbw=$LBW pw=$PW model=linear normalization=global local_norm=local_z use_wandb=True experiment=endo_exo,endo_only seed=0"
sbatch --export=ALL --job-name="${IEEE_RUNS}_local" -o "${IEEE_RUNS}_difference_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=lieee lbw=$LBW pw=$PW model=mole,msar,kalmanfilter,xgboost,gp,mlp,timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx normalization=global local_norm=difference use_wandb=True experiment=endo_exo,endo_only seed=0"
sbatch --export=ALL --job-name="${IEEE_RUNS}_difference" -o "${IEEE_RUNS}_difference_%j.out" --time="$TIME" --wrap="$JOB"
