#!/bin/bash
#SBATCH --account=def-qcappart
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --job-name=job_147
#SBATCH --output=knapsack/output/output_147_sg.log
#SBATCH --error=knapsack/output/error_147_sg.log
#SBATCH --time=00:26:40
module load cuda/12.2
module load python/3.10
source ../env_projet/bin/activate
python -m knapsack.run_experiments --ep 100000000 --patience 250 --id 147 --diff IMLE --lambd 10.0 --sigma 1.0 --n_samples 1 --n 100 --dim 10 --lr 0.01 --tl 600 --method SG --step_mu 50 --n_iter_mu 30 --loss 1