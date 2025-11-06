#!/bin/bash

# ressource specs
TIME=24:00:00

JOB="python main.py --multirun \
  hydra/launcher=cpu \
  dataset=ldalia,lwildppg \
  lbw=d \
  pw=a \
  model=linear,xgboost \
  normalization=global \
  local_norm=difference \
  use_wandb=True \
  feature=mean"
sbatch \
  --job-name="cpu" \
  -o "local_cpu_%j.out" \
  --time="$TIME" \
  --wrap="$JOB"


JOB="python main.py --multirun \
  hydra/launcher=gpu_small \
  dataset=ldalia,lwildppg \
  lbw=d \
  pw=a \
  model=mole,msar,kalmanfilter,gp,mlp \
  normalization=global \
  local_norm=difference \
  use_wandb=True \
  feature=mean"

sbatch \
  --job-name="gpu_small" \
  -o "local_gpu_small_%j.out" \
  --time="$TIME" \
  --wrap="$JOB"


JOB="python main.py --multirun \
  hydra/launcher=gpu_large \
  dataset=ldalia,lwildppg \
  lbw=d \
  pw=a \
  model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx \
  normalization=global \
  local_norm=local_z \
  use_wandb=True \
  feature=mean"

sbatch \
  --job-name="gpu_large" \
  -o "lbw_abl_gpu_l_%j.out" \
  --time="$TIME" \
  --wrap="$JOB"
