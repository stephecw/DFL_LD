#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=portfolio_classic_IMLE_n50_seed2
#SBATCH --output=logs/portfolio_classic_IMLE_n50_seed2.out
#SBATCH --error=logs/portfolio_classic_IMLE_n50_seed2.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 50 \
  --ep_cla 1000000 --lr 0.01 \
  --method IMLE \
  --seed 2 \
  --report 10 60 300 600 \
  --out_file portfolio/seed/results_seed2.csv \
  --time_limit 600 \
