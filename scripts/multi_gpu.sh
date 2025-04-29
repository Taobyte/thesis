# (multi_gpu.sh)
#!/bin/bash -l

# SLURM SUBMIT SCRIPT
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=24G
#SBATCH --mem=0
#SBATCH --time=0-02:00:00

module load eth_proxy

srun python main.py model=timellm num_workers=8 