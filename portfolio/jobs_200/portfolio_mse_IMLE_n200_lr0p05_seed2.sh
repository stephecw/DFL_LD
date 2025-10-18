#!/bin/bash
#SBATCH --time=34:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=portfolio_mse_IMLE_n200_lr0p05_seed2
#SBATCH --output=logs/portfolio_mse_IMLE_n200_lr0p05_seed2.out
#SBATCH --error=logs/portfolio_mse_IMLE_n200_lr0p05_seed2.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 200 --ep_mse 1000000 \
  --lr 0.05 \
  --method IMLE \
  --seed 2 \
  --report 10 60 300 600 1800 3600 7200 --num_eval_per_cp 100\
  --out_file portfolio/n50/results_seed2.csv \
  
