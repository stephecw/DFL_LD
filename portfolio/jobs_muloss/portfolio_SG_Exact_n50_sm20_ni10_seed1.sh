#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=portfolio_SG_Exact_n50_sm20_ni10_seed1
#SBATCH --output=logs/portfolio_SG_Exact_n50_sm20_ni10_seed1.out
#SBATCH --error=logs/portfolio_SG_Exact_n50_sm20_ni10_seed1.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 50 \
  --ep_sg 1000000 --step_mu 20 --n_iter_mu 10 \
  --method Exact \
  --seed 1 \
  --report 10 60 300 600 \
  --out_file portfolio/seed/results_seed1_muloss.csv \
  --time_limit 600 \
  --muloss 0
