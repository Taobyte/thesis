#!/bin/bash -l

GPUS=1
MODEL="$1"
DATASET="$2"
LBW="$3"
PW="$4"

echo $MODEL 
echo $DATASET 
echo $LBW 
echo $PW

torchrun --nproc_per_node=$GPUS \
    main.py \
    --multirun hydra/sweeper="$MODEL"_sweeper \
    model=$MODEL \
    dataset=$DATASET \
    look_back_window=$LBW \
    prediction_window=$PW \
    overfit=False \
    tune=True \ 
    use_wandb=False 



