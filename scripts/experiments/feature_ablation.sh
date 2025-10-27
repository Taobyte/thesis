#!/bin/bash

TIME="24:00:00"
OUT_PREFIX="wppg_feature_ablation"


CMD_CPU="python main.py --multirun \
  hydra/launcher=cpu \
  dataset=wildppg,dalia \
  lbw=c \
  pw=a \
  use_wandb=True \
  tune=False \
  normalization=global \
  folds=fold_0,fold_1,fold_2 \
  model=linear \
  local_norm=difference \
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
  local_norm=difference\
  folds=fold_0,fold_1,fold_2 \
  model=gp \
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
  folds=fold_0,fold_1,fold_2 \
  model=timesnet,simpletm \
  feature=rms_last2s_rms_jerk_centroid,catch22"

sbatch \
  --job-name="gpu_large" \
  --output="${OUT_PREFIX}_gpu_l_%j.out" \
  --time="${TIME}" \
  --wrap="bash -lc 'source ~/.bashrc && conda activate thesis && ${CMD_GPU_LARGE}'"
