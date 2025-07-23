#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=portfolio_SG_Exact_n200_sm10_ni10_lr0p001_seed0
#SBATCH --output=logs/portfolio_SG_Exact_n200_sm10_ni10_lr0p001_seed0.out
#SBATCH --error=logs/portfolio_SG_Exact_n200_sm10_ni10_lr0p001_seed0.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments   --n 50   --ep_sg 1000000 --step_mu 10 --n_iter_mu 10 --lr 0.001   --method Exact   --seed 0   --report 10 60 300 600 1800 3600   --out_file portfolio/n50/results_hyper_seed0.csv      --num_eval_per_cp 100
