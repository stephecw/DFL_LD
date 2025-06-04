#!/bin/bash
#SBATCH --account=def-qcappart
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=job_37
#SBATCH --output=knapsack/output/output_37.log
#SBATCH --error=knapsack/output/error_37.log
#SBATCH --time=09:00:00
module load cuda/12.2
module load python/3.10
source ../env_projet/bin/activate
python -m knapsack.run_experiments --ep 100000000 --tl 27600 --method LD --diff IMLE --dim 10 --n 200 --lr 0.1 --step_mu 0 --n_iter_mu 0