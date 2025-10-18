#!/bin/bash
#SBATCH --time=82:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=portfolio_mse_IMLE_n400_lr0p1_seed0
#SBATCH --output=logs/portfolio_mse_IMLE_n400_lr0p1_seed0.out
#SBATCH --error=logs/portfolio_mse_IMLE_n400_lr0p1_seed0.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
module load gurobi
export LC_ALL=C
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 400 --ep_mse 1000000 \
  --lr 0.1 \
  --method IMLE \
  --seed 0 \
  --report 10 60 300 600 1800 3600 7200 --num_eval_per_cp 100\
  --out_file portfolio/n400/results_seed0.csv \
  
