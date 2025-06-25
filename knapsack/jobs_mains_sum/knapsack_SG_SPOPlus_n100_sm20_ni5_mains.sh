#!/bin/bash
#SBATCH --time=02:30:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=knapsack_SG_SPOPlus_n100_sm20_ni5_mains
#SBATCH --output=logs/knapsack_SG_SPOPlus_n100_sm20_ni5_mains.out
#SBATCH --error=logs/knapsack_SG_SPOPlus_n100_sm20_ni5_mains.err

module load StdEnv/2023 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m knapsack.run_experiments \
  --diff SPOPlus \
  --method SG \
  --n 100 \
  --ep 1000000 \
  --tl 3600 \
  --step_mu 20 \
  --n_iter_mu 5 \
  --report 60 600 1800 3600 \
  --out_file knapsack/seed/results_seed0_mains_01.csv \
  --seed 0 \
--mains 0 1
