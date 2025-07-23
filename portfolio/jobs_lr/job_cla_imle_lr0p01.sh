#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=portfolio_cla_IMLE_n200_lr0p01_seed0
#SBATCH --output=logs/portfolio_cla_IMLE_n200_lr0p01_seed0.out
#SBATCH --error=logs/portfolio_cla_IMLE_n200_lr0p01_seed0.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments   --n 200   --ep_cla 1000000 --lr 0.01   --method IMLE   --seed 0   --report 10 60 300 600 1800 3600 7200   --out_file portfolio/n200/results_hyper_seed0.csv      --num_eval_per_cp 100
