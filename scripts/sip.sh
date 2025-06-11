#!/bin/bash

GPUS=1  # The gpu number
MODEL_NAME=pyramid_flux     # The model name, `pyramid_flux` or `pyramid_mmdit`
VAE_MODEL_PATH=/local/home/bdemirel/Projects/MultiScale_Vid/causal_video_vae  # The VAE CKPT dir.
ANNO_FILE=annotation/video_text.jsonl   # The video annotation file path
WIDTH=640
HEIGHT=384
NUM_FRAMES=121

export PYTHONPATH=$(pwd)/..

torchrun --nproc_per_node $GPUS \
    tools/extract_video_vae_latents.py \
    --batch_size 1 \
    --model_dtype bf16 \
    --model_path $VAE_MODEL_PATH \
    --anno_file $ANNO_FILE \
    --width $WIDTH \
    --height $HEIGHT \
    --num_frames $NUM_FRAMES