#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=portfolio_SG_SPOPlus_n200_sm20_ni5_lr0p01_seed0
#SBATCH --output=logs/portfolio_SG_SPOPlus_n200_sm20_ni5_lr0p01_seed0.out
#SBATCH --error=logs/portfolio_SG_SPOPlus_n200_sm20_ni5_lr0p01_seed0.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments   --n 200   --ep_sg 1000000 --step_mu 20 --n_iter_mu 5 --lr 0.01   --method SPOPlus   --seed 0   --report 10 60 300 600 1800 3600 7200   --out_file portfolio/n200/results_hyper_seed0.csv      --num_eval_per_cp 100
