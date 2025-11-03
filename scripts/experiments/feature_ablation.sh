#!/bin/bash

TIME="24:00:00"
OUT_PREFIX="feature_ablation"


CMD_CPU="python main.py --multirun \
  hydra/launcher=cpu \
  dataset=wildppg,dalia \
  lbw=d \
  pw=a \
  use_wandb=True \
  tune=False \
  normalization=global \
  folds=fold_0,fold_1,fold_2 \
  model=linear,xgboost \
  local_norm=difference \
  local_norm_endo_only=True \
  feature=rms_last2s_rms_jerk_centroid,catch22"

sbatch \
  --job-name="cpu" \
  --output="${OUT_PREFIX}_cpu_%j.out" \
  --time="${TIME}" \
  --wrap="bash -lc 'source ~/.bashrc && conda activate thesis && ${CMD_CPU}'"


CMD_GPU_SMALL="python main.py --multirun \
  hydra/launcher=gpu_small \
  dataset=wildppg,dalia \
  lbw=d \
  pw=a \
  use_wandb=True \
  tune=False \
  normalization=global \
  local_norm=difference \
  local_norm_endo_only=True \
  folds=fold_0,fold_1,fold_2 \
  model=mole,msar,kalmanfilter,gp,mlp \
  feature=rms_last2s_rms_jerk_centroid,catch22"

sbatch \
  --job-name="gpu_small" \
  --output="${OUT_PREFIX}_gpu_s_%j.out" \
  --time="${TIME}" \
  --wrap="bash -lc 'source ~/.bashrc && conda activate thesis && ${CMD_GPU_SMALL}'"




CMD_GPU_LARGE="python main.py --multirun \
  hydra/launcher=gpu_large \
  dataset=wildppg,dalia \
  lbw=d \
  pw=a \
  use_wandb=True \
  tune=False \
  normalization=global \
  local_norm=local_z \
  local_norm_endo_only=True \
  folds=fold_0,fold_1,fold_2 \
  model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx \
  feature=rms_last2s_rms_jerk_centroid,catch22"

sbatch \
  --job-name="gpu_large" \
  --output="${OUT_PREFIX}_gpu_l_%j.out" \
  --time="${TIME}" \
  --wrap="bash -lc 'source ~/.bashrc && conda activate thesis && ${CMD_GPU_LARGE}'"
