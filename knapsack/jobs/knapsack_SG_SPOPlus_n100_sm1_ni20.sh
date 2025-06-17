#!/bin/bash
#SBATCH --time=04:30:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=knapsack_SG_SPOPlus_n100_sm1_ni20
#SBATCH --output=logs/knapsack_SG_SPOPlus_n100_sm1_ni20.out
#SBATCH --error=logs/knapsack_SG_SPOPlus_n100_sm1_ni20.err

module load StdEnv/2023 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m knapsack.run_experiments \
  --diff SPOPlus \
  --method SG \
  --n 100 \
  --ep 1000000 \
  --tl 3600 \
  --step_mu 1 \
  --n_iter_mu 20 \
  --report 60 600 1800 3600 \
  --out_file knapsack/results_deg8_mains1.csv \
  --mains 1 \

