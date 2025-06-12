#!/bin/bash -l

GPUS=1

torchrun --nproc_per_node=$GPUS \
    main.py \
    --multirun hydra/sweeper=${1}_sweeper \
    model=${1} \
    dataset=${2} \ 
    look_back_window=${3} \ 
    prediction_window=${4} \ 
    tune=True \ 
    overfit=False 



