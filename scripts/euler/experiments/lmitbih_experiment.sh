#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

PW=${1} #  f = 30 for lmitbih = 15s i = 75 for 45s

# ressource specs
TIME=24:00:00

BLNNAME="lmitbih_baselines_cpu"
JOB="python main.py --multirun hydra/launcher=cpu dataset=lmitbih lbw=lbw pw=$PW model=linear,xgboost normalization=difference use_wandb=True experiment=lbw_mitbih seed=0,1,2"
sbatch --job-name="$BLNNAME" -o "$BLNNAME" --time="$TIME" --wrap="$JOB"

BLNNAME="lmitbih_baselines_gpu"
JOB="python main.py --multirun hydra/launcher=gpu_small dataset=lmitbih lbw=lbw pw=$PW model=mole,msar,kalmanfilter,gp,mlp normalization=difference use_wandb=True experiment=lbw_mitbih seed=0,1,2"
sbatch --job-name="$BLNNAME" -o "$BLNNAME" --time="$TIME" --wrap="$JOB"

BLGNAME="lmitbih_dls"
JOB="python main.py --multirun hydra/launcher=slurm_eff dataset=lmitbih lbw=lbw pw=$PW model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx use_wandb=True experiment=lbw_mitbih seed=0,1,2"
sbatch --job-name="$BLGNAME" -o "$BLGNAME" --time="$TIME" --wrap="$JOB"
