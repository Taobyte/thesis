MODEL=${1}
NORMALIZATION=${2}
N_TRIALS=${3}
DATASET=${4}
GPU_MEM=${5}
LBWS=(a b c d f) # 5, 10, 20, 30, 3 -> we do not use 60 for local tuning
PW=a

for LBW in "${LBWS[@]}"
do
    sbatch scripts/euler/tuning/single_lbw_tune.sh "$MODEL" "$NORMALIZATION" "$N_TRIALS" "$LBW" "$PW" "$DATASET" "$GPU_MEM"
done


