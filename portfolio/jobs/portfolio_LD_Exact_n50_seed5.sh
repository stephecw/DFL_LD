#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=portfolio_LD_Exact_n50_seed5
#SBATCH --output=logs/portfolio_LD_Exact_n50_seed5.out
#SBATCH --error=logs/portfolio_LD_Exact_n50_seed5.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 50 \
  --ep_ld 1000000 \
  --method Exact \
  --seed 5 \
  --report 10 60 300 600 \
  --out_file portfolio/seed/results_seed5.csv \
  --time_limit 600 \
