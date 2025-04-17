#!/bin/bash

# Array of values
look_back_windows=(10 15 20)
prediction_windows=(1 2 3)

NUM_WORKERS=8
MEM_PER_CPU=4G
TIME=24:00:00

mkdir -p logs

# Loop through all combinations
for lbw in "${look_back_windows[@]}"; do
  for pw in "${prediction_windows[@]}"; do
    JOB="python main.py +look_back_window=$lbw +prediction_window=$pw"
    sbatch \
      --job-name=lbw${lbw}_pw${pw} \
      --output=logs/lbw${lbw}_pw${pw}_%j.out \
      --cpus-per-task="$NUM_WORKERS" \
      --mem-per-cpu="$MEM_PER_CPU" \
      --time="$TIME" \
      --gres=gpu:1 \
      --wrap="nvidia-smi; $JOB"
  done
done
