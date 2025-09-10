#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

# ressource specs
TIME=24:00:00
LBW=d # d = 30 
PW=a # a = 3

# ---- Deep models (local CV) ----
DNAME="LOCAL_RUNS_DL"
JOB="python main.py --multirun \
  hydra/launcher=submitit_slurm \
  dataset=ldalia,lwildppg,lieee \
  lbw=${LBW} pw=${PW} \
  model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx \
  normalization=global local_norm=local_z use_wandb=True tune=False experiment=endo_exo"

sbatch --job-name="$DNAME" -o "${DNAME}_%j.out" --time="$TIME" --wrap="$JOB"

# ---- Baselines (local CV) ----
BNAME="LOCAL_RUNS_BASELINES"
JOB="python main.py --multirun \
  hydra/launcher=submitit_slurm \
  dataset=ldalia,lwildppg,lieee \
  lbw=${LBW} pw=${PW} \
  model=linear,mole,msar,kalmanfilter,gp,xgboost,mlp \
  normalization=global local_norm=difference use_wandb=True tune=False experiment=endo_exo"

sbatch --job-name="$BNAME" -o "${BNAME}_%j.out" --time="$TIME" --wrap="$JOB"
