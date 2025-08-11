MODEL=${1}
NORMALIZATION=${2}
N_TRIALS=${3}
LBWS=(5 10 20 30 60)
for lbw in "${LBWS[@]}"
do
	sbatch scripts/euler/single_lbw_tune.sh "$MODEL" "$NORMALIZATION" "$N_TRIALS" "$lbw" 3
done
