#!/bin/bash
#SBATCH --account=def-qcappart
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=job_11
#SBATCH --output=knapsack/output/output_11.log
#SBATCH --error=knapsack/output/error_11.log
#SBATCH --time=05:00:00
module load cuda/12.2
module load python/3.10
source ../env_projet/bin/activate
python -m knapsack.run_experiments --ep 100000000 --tl 13200 --method MSE --diff IMLE --dim 5 --n 100 --lr 0.001 --step_mu 0 --n_iter_mu 0