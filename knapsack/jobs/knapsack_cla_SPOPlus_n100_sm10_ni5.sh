#!/bin/bash
#SBATCH --time=04:30:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=knapsack_cla_SPOPlus_n100_sm10_ni5
#SBATCH --output=logs/knapsack_cla_SPOPlus_n100_sm10_ni5.out
#SBATCH --error=logs/knapsack_cla_SPOPlus_n100_sm10_ni5.err

module load StdEnv/2023 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m knapsack.run_experiments \
  --diff SPOPlus \
  --method cla \
  --n 100 \
  --ep 1000000 \
  --tl 3600 \
  --step_mu 10 \
  --n_iter_mu 5 \
  --report 60 600 1800 3600 \
  --out_file knapsack/seed/results_seed0.csv \
  --seed 0 \

