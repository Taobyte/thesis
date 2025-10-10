#!/bin/bash -l

# Static resources
GPUS=1
N_CPUS=1
CPU_MEM="4G"

# Parameters
MODEL="$1"
NORMALIZATION="$2"
N_TRIALS="$3"
LBW="$4"
PW="$5"
DATASET="$6"
GPU_MEM="$7"    
TIME="$8"

NAME="optuna_${MODEL}_${DATASET}_${NORMALIZATION}_${LBW}_${PW}"
LOGDIR="/cluster/project/holz/ckeusch/tune_logs"

JOB="python main.py --multirun \
  hydra/sweeper=${MODEL}_sweeper \
  model=${MODEL} \
  normalization=${NORMALIZATION} \
  experiment=endo_exo \
  n_trials=${N_TRIALS} \
  lbw=${LBW} \
  pw=${PW} \
  dataset=${DATASET} \
  tune=True \
  overfit=False"

sbatch \
  --job-name "$NAME" \
  --output "${LOGDIR}/${NAME}_%j.out" \
  --cpus-per-task "$N_CPUS" \
  --mem-per-cpu "$CPU_MEM" \
  --time "$TIME" \
  --gpus "$GPUS" \
  --gres "gpumem:${GPU_MEM}G" \
  --wrap "bash -lc 'source ~/.bashrc; conda activate thesis; module load eth_proxy; nvidia-smi; $JOB'"
