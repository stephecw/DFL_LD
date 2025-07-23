#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=portfolio_cla_SPOPlus_n50_lr0p005_seed6
#SBATCH --output=logs/portfolio_cla_SPOPlus_n50_lr0p005_seed6.out
#SBATCH --error=logs/portfolio_cla_SPOPlus_n50_lr0p005_seed6.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 50 --ep_cla 1000000 \
  --lr 0.005 \
  --method SPOPlus \
  --seed 6 \
  --report 10 60 300 600 1800 3600 --num_eval_per_cp 500\
  --out_file portfolio/n50/results_seed6.csv \
  
