#!/bin/bash

TIME="24:00:00"
OUT_PREFIX="global_norm_exp"


CMD_CPU="python main.py --multirun \
  hydra/launcher=cpu \
  dataset=dalia,wildppg,ieee \
  lbw=d \
  pw=a \
  use_wandb=True \
  normalization=global \
  folds=fold_0,fold_1,fold_2 \
  model=linear,xgboost \
  local_norm=lnone \
  feature=mean"

sbatch \
  --job-name="cpu" \
  --output="${OUT_PREFIX}_cpu_%j.out" \
  --time="${TIME}" \
  --wrap="$CMD_CPU"


CMD_GPU_SMALL="python main.py --multirun \
  hydra/launcher=gpu_small \
  dataset=dalia,wildppg,ieee \
  lbw=d \
  pw=a \
  use_wandb=True \
  normalization=global \
  folds=fold_0,fold_1,fold_2 \
  model=mole,msar,kalmanfilter,gp,mlp \
  local_norm=lnone \
  feature=mean"

sbatch \
  --job-name="gpu_small" \
  --output="${OUT_PREFIX}_gpu_s_%j.out" \
  --time="${TIME}" \
  --wrap="$CMD_GPU_SMALL"


CMD_GPU_LARGE="python main.py --multirun \
  hydra/launcher=gpu_large \
  dataset=dalia,wildppg,ieee \
  lbw=d \
  pw=a \
  use_wandb=True \
  normalization=global \
  folds=fold_0,fold_1,fold_2 \
  model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx \
  local_norm=lnone \
  feature=mean"

sbatch \
  --job-name="gpu_large" \
  --output="${OUT_PREFIX}_gpu_l_%j.out" \
  --time="${TIME}" \
  --wrap="$CMD_GPU_LARGE"


# Local Normalization Experiments

CMD_CPU="python main.py --multirun \
  hydra/launcher=cpu \
  dataset=dalia,wildppg,ieee \
  lbw=d \
  pw=a \
  use_wandb=True \
  normalization=global \
  folds=fold_0,fold_1,fold_2 \
  model=linear,xgboost \
  local_norm=difference,local_z \
  local_norm_endo_only=True,False \
  feature=mean"

sbatch \
  --job-name="cpu" \
  --output="${OUT_PREFIX}_cpu_%j.out" \
  --time="${TIME}" \
  --wrap="$CMD_CPU"


CMD_GPU_SMALL="python main.py --multirun \
  hydra/launcher=gpu_small \
  dataset=dalia,wildppg,ieee \
  lbw=d \
  pw=a \
  use_wandb=True \
  normalization=global \
  folds=fold_0,fold_1,fold_2 \
  model=mole,msar,kalmanfilter,gp,mlp \
  local_norm=difference,local_z \
  local_norm_endo_only=True,False \
  feature=mean"

sbatch \
  --job-name="gpu_small" \
  --output="${OUT_PREFIX}_gpu_s_%j.out" \
  --time="${TIME}" \
  --wrap="$CMD_GPU_SMALL"


CMD_GPU_LARGE="python main.py --multirun \
  hydra/launcher=gpu_large \
  dataset=dalia,wildppg,ieee \
  lbw=d \
  pw=a \
  use_wandb=True \
  normalization=global \
  folds=fold_0,fold_1,fold_2 \
  model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx \
  local_norm=difference,local_z \
  local_norm_endo_only=True,False \
  feature=mean"

sbatch \
  --job-name="gpu_large" \
  --output="${OUT_PREFIX}_gpu_l_%j.out" \
  --time="${TIME}" \
  --wrap="$CMD_GPU_LARGE"



