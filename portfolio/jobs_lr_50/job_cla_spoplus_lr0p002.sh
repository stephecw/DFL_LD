#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name=portfolio_cla_SPOPlus_n50_lr0p002_seed0
#SBATCH --output=logs/portfolio_cla_SPOPlus_n50_lr0p002_seed0.out
#SBATCH --error=logs/portfolio_cla_SPOPlus_n50_lr0p002_seed0.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments   --n 50   --ep_cla 1000000 --lr 0.002   --method SPOPlus   --seed 0   --report 10 60 300 600 1800 3600   --out_file portfolio/n50/results_hyper_seed0.csv  
