#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=portfolio_mse_SPOPlus_lr0p01
#SBATCH --output=logs/portfolio_mse_SPOPlus_lr0p01.out
#SBATCH --error=logs/portfolio_mse_SPOPlus_lr0p01.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 50 \
  --ep_mse 1000000 \
  --method SPOPlus \
  --seed 0 \
  --report 10 60 300 600 \
  --out_file portfolio/seed/results_lr.csv \
  --time_limit 600 \
  --lr 0.01
