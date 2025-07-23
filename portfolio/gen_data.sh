#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --job-name=gen_data
#SBATCH --output=logs/gen_data.out
#SBATCH --error=logs/gen_data.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
module load gurobi
source ~/env_projet/bin/activate

export LC_ALL=C

python -m portfolio.gen_data \
  --n 500 \
  --n_iter 500 \