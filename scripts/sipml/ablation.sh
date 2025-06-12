#!/bin/bash

GPUS=1  # Number of GPUs
DATASET="$1"
MODELS=("linear" "kalmanfilter" "gp" "timesnet" "simpletm" "pattn" "gpt4ts")

export PYTHONPATH=$(pwd)/..

 for model in "${MODELS[@]}"; do 
    echo "Start training ${model}"
    for lbw in {3,4,5}; do 
        for pw in {1}; do 
            for fold in {0,1,2}; do 
                FOLD_STR="fold_${fold}"
                torchrun --nproc_per_node=$GPUS main.py model=$model dataset=$DATASET look_back_window=$lbw prediction_window=$pw experiment=$FOLD_STR overfit=False use_wandb=True
        done 
    done
    echo "Finished training ${model}"
done