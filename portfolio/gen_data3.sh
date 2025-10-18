#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --job-name=gen_data3
#SBATCH --output=logs/gen_data3.out
#SBATCH --error=logs/gen_data3.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
module load gurobi
source ~/env_projet/bin/activate

export LC_ALL=C

# Lancement du script
python -m portfolio.changement_data \
  --fname portfolio/datasets/train_400_100_5_8_2-25.txt \
  --iters 9 \
  --principal_lin 0