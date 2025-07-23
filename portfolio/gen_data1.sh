#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --job-name=gen_data2
#SBATCH --output=logs/gen_data2.out
#SBATCH --error=logs/gen_data2.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
module load gurobi
source ~/env_projet/bin/activate

export LC_ALL=C

python -m portfolio.gen_data \
  --n 500 \
  --n_iter 300 \