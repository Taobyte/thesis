MODEL=${1}
NORMALIZATION=${2}
LBWS=(5 10 20 30 60)
for lbw in "${LBWS[@]}"
do
	sbatch scripts/euler/tune.sh "$MODEL" "$NORMALIZATION" "$lbw" 3
done
