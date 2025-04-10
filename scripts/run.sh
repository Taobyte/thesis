#!/bin/bash 


# ressource specs
NUM_WORKERS=8
TIME=04:00:00
MEM_PER_CPU=8G

JOB="python main.py path=cluster"
sbatch -n 1 --cpus-per-task "$NUM_WORKERS" --mem-per-cpu="$MEM_PER_CPU" --time="$TIME" -p gpu --gpus=1 --gres=gpumem:"$MEM_PER_GPU" --wrap="nvidia-smi;$JOB"
