N_TRIALS=50
PREDICTION_WINDOW=3
BASELINE_NONE=(linear hlinear kalmanfilter xgboost hxgboost)
BASELINE_DL_GLOBAL=(gp mlp timesnet simpletm adamshyper patchtst timexer gpt4ts)

for MODEL in "${BASELINE_NONE[@]}"
do
	sbatch scripts/euler/look_back_tune.sh "$MODEL" best_endo_exo none "$N_TRIALS" "$PREDICTION_WINDOW"
done

for MODEL in "${BASELINE_DL_GLOBAL[@]}"
do
	sbatch scripts/euler/look_back_tune.sh "$MODEL" best_endo_exo global "$N_TRIALS" "$PREDICTION_WINDOW"
done
