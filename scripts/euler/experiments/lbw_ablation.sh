#!/bin/bash 

# setup environment 
source ~/.bashrc
conda activate thesis

# ressource specs
TIME=24:00:00


# DALIA 

DALIA_RUNS="lbw_dalia_ablation"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia lbw=a,b,c,d,e pw=a model=linear,kalmanfilter,gp,adamshyper,nbeatsx normalization=global use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$DALIA_RUNS" -o "$DALIA_RUNS" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia lbw=a,b,c,d,e pw=a model=timesnet,patchtst,timexer,gpt4ts normalization=none use_norm_dl=True use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$DALIA_RUNS" -o "$DALIA_RUNS" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=dalia lbw=a,b,c,d,e pw=a model=mole,msar,xgboost,mlp,simpletm normalization=difference use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$DALIA_RUNS" -o "$DALIA_RUNS" --time="$TIME" --wrap="$JOB"

# WILDPPG

WILDPPG_RUNS="lbw_wildppg_ablation"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=wildppg lbw=a,b,c,d,e pw=a model=nbeatsx normalization=global use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$WILDPPG_RUNS" -o "$WILDPPG_RUNS" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=wildppg lbw=a,b,c,d,e pw=a model=mole,xgboost,mlp,timesnet,simpletm,adamshyper,patchtst,timexer normalization=global use_norm_dl=True use_norm_baseline=True use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$WILDPPG_RUNS" -o "$WILDPPG_RUNS" --time="$TIME" --wrap="$JOB"

JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=wildppg lbw=a,b,c,d,e pw=a model=linear,msar,kalmanfilter,gp,gpt4ts normalization=difference use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$WILDPPG_RUNS" -o "$WILDPPG_RUNS" --time="$TIME" --wrap="$JOB"

# IEEE

IEEE_RUNS="lbw_ieee_ablation"
JOB="python main.py --multirun hydra/launcher=submitit_slurm dataset=ieee lbw=a,b,c,d,e pw=a model=linear,mole,msar,kalmanfilter,xgboost,gp,mlp,timesnet,simpletm,adamshyper,patchtst,timexer,gpt4ts,nbeatsx normalization=difference use_wandb=True tune=False experiment=ablation folds=fold_0,fold_1,fold_2"
sbatch --job-name="$IEEE_RUNS" -o "$IEEE_RUNS" --time="$TIME" --wrap="$JOB"
