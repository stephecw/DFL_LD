#!/bin/bash
#SBATCH --time=34:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=portfolio_LD_Exact_n200_lr1p0_seed3
#SBATCH --output=logs/portfolio_LD_Exact_n200_lr1p0_seed3.out
#SBATCH --error=logs/portfolio_LD_Exact_n200_lr1p0_seed3.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 200 --ep_ld 1000000 \
  --lr 1.0 \
  --method Exact \
  --seed 3 \
  --report 10 60 300 600 1800 3600 7200 --num_eval_per_cp 100\
  --out_file portfolio/n50/results_seed3.csv \
  --muloss 0
