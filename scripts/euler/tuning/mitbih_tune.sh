#!/bin/bash -l

# Static resources
GPUS=1
N_CPUS=1
CPU_MEM="4G"
BASELINES=(linear mole msar kalmanfilter xgboost gp mlp)
DLS=(timesnet simpletm adamshyper patchtst timexer gpt4ts nbeatsx)

# Parameters
N_TRIALS=100
PW=f # f = 30 
GPU_MEM=24
TIME=24:00:00

LOGDIR="/cluster/project/holz/ckeusch/tune_logs"

for MODEL in "${BASELINES[@]}"
do 
    NAME="BL_${MODEL}_mitbih_tune"
    JOB="python main.py --multirun \
    hydra/sweeper=${MODEL}_sweeper \
    model=${MODEL} \
    normalization=difference \
    experiment=lbw_mitbih \
    n_trials=${N_TRIALS} \
    pw=${PW} \
    dataset=lmitbih \
    tune=True \
    overfit=False \
    +sweeps=local_lbw"

    sbatch \
    --job-name "$NAME" \
    --output "${LOGDIR}/${NAME}_%j.out" \
    --cpus-per-task "$N_CPUS" \
    --mem-per-cpu "$CPU_MEM" \
    --time "$TIME" \
    --gpus "$GPUS" \
    --gres "gpumem:${GPU_MEM}G" \
    --wrap "bash -lc 'source ~/.bashrc; conda activate thesis; module load eth_proxy; nvidia-smi; $JOB'"
done


for MODEL in "${DLS[@]}"
do 
    NAME="DL_${MODEL}_mitbih_tune"
    JOB="python main.py --multirun \
    hydra/sweeper=${MODEL}_sweeper \
    model=${MODEL} \
    experiment=lbw_mitbih \
    n_trials=${N_TRIALS} \
    pw=${PW} \
    dataset=lmitbih \
    tune=True \
    overfit=False \
    +sweeps=local_lbw"

    sbatch \
    --job-name "$NAME" \
    --output "${LOGDIR}/${NAME}_%j.out" \
    --cpus-per-task "$N_CPUS" \
    --mem-per-cpu "$CPU_MEM" \
    --time "$TIME" \
    --gpus "$GPUS" \
    --gres "gpumem:${GPU_MEM}G" \
    --wrap "bash -lc 'source ~/.bashrc; conda activate thesis; module load eth_proxy; nvidia-smi; $JOB'"
done