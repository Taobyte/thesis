#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

# ressource specs
TIME=24:00:00


# GLOBAL RUNS
JOB="python main.py --multirun hydra/launcher=cpu test_local=True dataset=dalia,wildppg,ieee lbw=b pw=a,b,c,d model=linear,xgboost normalization=global local_norm=difference use_wandb=True experiment=endo_exo,endo_only seed=1 folds=fold_0,fold_1,fold_2"
sbatch --job-name="cpu" -o "lbw_abl_cpu_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=gpu_small test_local=True dataset=dalia,wildppg,ieee lbw=b pw=a,b,c,d model=mole,msar,kalmanfilter,gp,mlp normalization=global local_norm=difference use_wandb=True experiment=endo_exo,endo_only seed=1 folds=fold_0,fold_1,fold_2"
sbatch --job-name="gpu_small" -o "lbw_abl_gpu_s_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=slurm_eff test_local=True dataset=dalia,wildppg,ieee lbw=b pw=a,b,c,d model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx normalization=global local_norm=local_z use_wandb=True experiment=endo_exo,endo_only seed=1 folds=fold_0,fold_1,fold_2"
sbatch --job-name="gpu_large" -o "lbw_abl_gpu_l_%j.out" --time="$TIME" --wrap="$JOB"


# LOCAL RUNS
JOB="python main.py --multirun hydra/launcher=cpu dataset=ldalia,lwildppg,lieee lbw=b pw=a,b,c,d model=linear,xgboost normalization=global local_norm=difference use_wandb=True experiment=endo_exo,endo_only seed=1"
sbatch --job-name="cpu" -o "lbw_abl_cpu_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=gpu_small dataset=ldalia,lwildppg,lieee lbw=b pw=a,b,c,d model=mole,msar,kalmanfilter,gp,mlp normalization=global local_norm=difference use_wandb=True experiment=endo_exo,endo_only seed=1"
sbatch --job-name="gpu_small" -o "lbw_abl_gpu_s_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=slurm_eff dataset=ldalia,lwildppg,lieee lbw=b pw=a,b,c,d model=timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx normalization=global local_norm=local_z use_wandb=True experiment=endo_exo,endo_only seed=1"
sbatch --job-name="gpu_large" -o "lbw_abl_gpu_l_%j.out" --time="$TIME" --wrap="$JOB"


