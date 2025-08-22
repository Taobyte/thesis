MODEL=${1}
NORMALIZATION=${2}
N_TRIALS=${3}
LBWS=(a b c d e)
for lbw in "${LBWS[@]}"
do
	sbatch scripts/euler/tuning/single_lbw_tune.sh "$MODEL" "$NORMALIZATION" "$N_TRIALS" "$lbw" a
done
