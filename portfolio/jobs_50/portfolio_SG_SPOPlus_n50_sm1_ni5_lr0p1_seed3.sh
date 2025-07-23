#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=portfolio_SG_SPOPlus_n50_sm1_ni5_lr0p1_seed3
#SBATCH --output=logs/portfolio_SG_SPOPlus_n50_sm1_ni5_lr0p1_seed3.out
#SBATCH --error=logs/portfolio_SG_SPOPlus_n50_sm1_ni5_lr0p1_seed3.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 50 --ep_sg 1000000 --step_mu 1 --n_iter_mu 5 \
  --lr 0.1 \
  --method SPOPlus \
  --seed 3 \
  --report 10 60 300 600 1800 3600 --num_eval_per_cp 500\
  --out_file portfolio/n50/results_seed3.csv \
  
