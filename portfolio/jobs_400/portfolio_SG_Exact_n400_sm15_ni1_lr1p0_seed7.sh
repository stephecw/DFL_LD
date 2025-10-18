#!/bin/bash
#SBATCH --time=82:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=portfolio_SG_Exact_n400_sm15_ni1_lr1p0_seed7
#SBATCH --output=logs/portfolio_SG_Exact_n400_sm15_ni1_lr1p0_seed7.out
#SBATCH --error=logs/portfolio_SG_Exact_n400_sm15_ni1_lr1p0_seed7.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
module load gurobi
export LC_ALL=C
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 400 --ep_sg 1000000 --step_mu 15 --n_iter_mu 1 \
  --lr 1.0 \
  --method Exact \
  --seed 7 \
  --report 10 60 300 600 1800 3600 7200 --num_eval_per_cp 100\
  --out_file portfolio/n400/results_seed7.csv \
  --muloss 0
