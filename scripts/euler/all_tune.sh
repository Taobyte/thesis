MODEL=${1}
NORMALIZATION=${2}
LBWS=(5 10 20 30 60)
DYNAMIC=(True False)
for b in "${DYNAMIC[@]}"
do
	for lbw in "${LBWS[@]}"
	do
		sbatch scripts/euler/tune.sh "$MODEL" "$NORMALIZATION" "$b" "$lbw" 3
	done
done
