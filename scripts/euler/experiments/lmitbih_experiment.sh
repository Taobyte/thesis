#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

PW=${1} #  f = 30 for lmitbih = 15s

# ressource specs
TIME=24:00:00

BLNNAME="lmitbih_baselines"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=lmitbih lbw=lbw pw=$PW model=linear,mole,msar,kalmanfilter,xgboost,gp,mlp normalization=difference use_wandb=True experiment=lbw_mitbih seed=0,1,2"
sbatch --job-name="$BLNNAME" -o "$BLNNAME" --time="$TIME" --wrap="$JOB"

BLGNAME="lmitbih_dls"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=lmitbih lbw=lbw pw=$PW model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx use_wandb=True experiment=lbw_mitbih seed=0,1,2"
sbatch --job-name="$BLGNAME" -o "$BLGNAME" --time="$TIME" --wrap="$JOB"
