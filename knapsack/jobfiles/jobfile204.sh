#!/bin/bash
#SBATCH --account=def-qcappart
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --job-name=job_204
#SBATCH --output=knapsack/output/output_204_sg.log
#SBATCH --error=knapsack/output/error_204_sg.log
#SBATCH --time=00:17:10
module load cuda/12.2
module load python/3.10
source ../env_projet/bin/activate
python -m knapsack.run_experiments --ep 100000000 --patience 5 --id 204 --diff IMLE --lambd 10.0 --sigma 1.0 --n_samples 1 --n 100 --dim 10 --lr 0.1 --tl 30 --method SG --step_mu 1 --n_iter_mu 50