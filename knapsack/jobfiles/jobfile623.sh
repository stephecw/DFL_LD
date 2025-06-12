#!/bin/bash
#SBATCH --account=def-qcappart
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3
#SBATCH --job-name=job_623
#SBATCH --output=knapsack/output/output_623_sg.log
#SBATCH --error=knapsack/output/error_623_sg.log
#SBATCH --time=02:16:40
module load cuda/12.2
module load python/3.10
source ../env_projet/bin/activate
python -m knapsack.run_experiments --ep 100000000 --patience 150 --id 623 --diff SPOPlus --n 100 --dim 10 --lr 0.1 --tl 7200 --method SG --step_mu 30 --n_iter_mu 10