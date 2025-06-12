#!/bin/bash

GPUS=1  # Number of GPUs
DATASET="$1"
MODELS=("linear" "kalmanfilter" "gp" "timesnet" "simpletm" "pattn" "adamshyper")

export PYTHONPATH=$(pwd)/..

for lbw in {3,..,5}; do 
    for pw in {1,..,3}; do 
        for model in $MODELS; do 
            echo "$lbw $pw $model"



`
torchrun --nproc_per_node=$GPUS \
    main.py \
    model="$MODEL" \
    dataset="$DATASET" \
    look_back_window="$LBW" \
    prediction_window="$PW" \
    overfit=False \
    use_wandb=True
`