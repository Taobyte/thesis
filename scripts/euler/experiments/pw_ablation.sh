#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

# ressource specs
TIME=24:00:00


# DALIA 

DALIA_RUNS="pw_dalia_ablation"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia lbw=d pw=b,c,d,e model=linear,kalmanfilter,gp,adamshyper,nbeatsx normalization=global use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$DALIA_RUNS" -o "$DALIA_RUNS_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia lbw=d pw=b,c,d,e model=timesnet,patchtst,timexer,gpt4ts normalization=none use_norm_dl=True use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$DALIA_RUNS" -o "$DALIA_RUNS_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia lbw=d pw=b,c,d,e model=mole,msar,xgboost,mlp,simpletm normalization=difference use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$DALIA_RUNS" -o "$DALIA_RUNS_%j.out" --time="$TIME" --wrap="$JOB"

# WILDPPG

WILDPPG_RUNS="pw_wildppg_ablation"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=wildppg lbw=d pw=b,c,d,e model=nbeatsx normalization=global use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$WILDPPG_RUNS_global" -o "$WILDPPG_RUNS_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=wildppg lbw=d pw=b,c,d,e model=mole,xgboost,mlp,timesnet,simpletm,adamshyper,patchtst,timexer normalization=global use_norm_dl=True use_norm_baseline=True use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$WILDPPG_RUNS_local" -o "$WILDPPG_RUNS_%j.out" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=wildppg lbw=d pw=b,c,d,e model=linear,msar,kalmanfilter,gp,gpt4ts normalization=difference use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$WILDPPG_RUNS_difference" -o "$WILDPPG_RUNS_%j.out" --time="$TIME" --wrap="$JOB"

# IEEE

IEEE_RUNS="pw_ieee_ablation"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=ieee lbw=d pw=b,c,d,e model=linear,mole,msar,kalmanfilter,xgboost,gp,mlp,timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx normalization=difference use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$IEEE_RUNS_difference" -o "$IEEE_RUNS_%j.out" --time="$TIME" --wrap="$JOB"
