#!/bin/bash
#SBATCH --time=80:00:00
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
  --n 400 \
  --n_iter 300 \