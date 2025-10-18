#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=portfolio_SG_Exact_n50_sm15_ni5_lr0p1_seed2
#SBATCH --output=logs/portfolio_SG_Exact_n50_sm15_ni5_lr0p1_seed2.out
#SBATCH --error=logs/portfolio_SG_Exact_n50_sm15_ni5_lr0p1_seed2.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 50 --ep_sg 1000000 --step_mu 15 --n_iter_mu 5 \
  --lr 0.1 \
  --method Exact \
  --seed 2 \
  --report 10 60 300 600 1800 3600 --num_eval_per_cp 500\
  --out_file portfolio/n50/results_sg_seed2 \
  --muloss 0
