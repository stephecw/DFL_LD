#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --job-name=timings
#SBATCH --output=logs/timings.out
#SBATCH --error=logs/timings.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
module load gurobi
source ~/env_projet/bin/activate

# export LC_ALL=C

python -m portfolio.bench_timings \
  --fname portfolio/datasets/train_50_100_5_8_2-25.txt \
  --iters 100 \
  --principal_lin 0 \
  --csv portfolio/timings_results.csv