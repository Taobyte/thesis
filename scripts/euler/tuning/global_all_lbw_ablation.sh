MODEL=${1}
NORMALIZATION=${2}
N_TRIALS=${3}
LBWS=(a b c d e) 
for LBW in "${LBWS[@]}"
do
	sbatch scripts/euler/tuning/single_lbw_tune.sh "$MODEL" "$NORMALIZATION" "$N_TRIALS" "$LBW" a
done
