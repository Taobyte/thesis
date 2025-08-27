
N_TRIALS=50
BASELINES=(xgboost mole msar kalmanfilter exactgp mlp) # we do not have to tune the linear model
DL=(timesnet simpletm adamshyper patchtst timexer gpt4ts nbeatsx)
DATASETS=(ldalia lwildppg lieee)
GPU_MEM_BASELINE=4
GPU_MEM_DL=24
TIME=24:00:00

for DATASET in "${DATASETS[@]}"
do
    for MODEL in "${BASELINES[@]}"
    do
        bash scripts/euler/tuning/local_lbw_ablation.sh "$MODEL" difference "$N_TRIALS" "$DATASET" "$GPU_MEM_BASELINE" "$TIME"
    done

    for MODEL in "${DL[@]}"
    do
        bash scripts/euler/tuning/local_lbw_ablation.sh "$MODEL" global "$N_TRIALS" "$DATASET" "$GPU_MEM_DL" "$TIME"
    done
done
