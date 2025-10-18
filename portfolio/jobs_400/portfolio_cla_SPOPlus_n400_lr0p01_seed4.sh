#!/bin/bash
#SBATCH --time=82:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=portfolio_cla_SPOPlus_n400_lr0p01_seed4
#SBATCH --output=logs/portfolio_cla_SPOPlus_n400_lr0p01_seed4.out
#SBATCH --error=logs/portfolio_cla_SPOPlus_n400_lr0p01_seed4.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
module load gurobi
export LC_ALL=C
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 400 --ep_cla 1000000 \
  --lr 0.01 \
  --method SPOPlus \
  --seed 4 \
  --report 10 60 300 600 1800 3600 7200 --num_eval_per_cp 100\
  --out_file portfolio/n400/results_seed4.csv \
  
