#!/bin/bash

GPUS=1  # Number of GPUs
MODEL="$1"
DATASET="$2"
LBW="$3"
PW="$4"

export PYTHONPATH=$(pwd)/..
export HYDRA_FULL_ERROR=1

torchrun --nproc_per_node=$GPUS \
    main.py \
    model="$MODEL" \
    dataset="$DATASET" \
    look_back_window="$LBW" \
    prediction_window="$PW" \
    overfit=False \
    use_wandb=True
