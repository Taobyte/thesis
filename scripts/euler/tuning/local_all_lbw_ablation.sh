
N_TRIALS=50
BASELINES=(linear xgboost mole msar kalmanfilter exactgp mlp)
DL=(timesnet simpletm adamshyper patchtst timexer gpt4ts nbeatsx)
DATASETS=(ldalia lwildppg lieee)
for DATASET in "${DATASETS[@]}"
do
    for MODEL in "${BASELINES[@]}"
    do
        bash scripts/euler/tuning/local_lbw_ablation_tune.sh "$MODEL" difference "$N_TRIALS" "$DATASET"
    done

    for MODEL in "${DL[@]}"
    do
        bash scripts/euler/tuning/local_lbw_ablation_tune.sh "$MODEL" global "$N_TRIALS" "$DATASET"
    done
done
