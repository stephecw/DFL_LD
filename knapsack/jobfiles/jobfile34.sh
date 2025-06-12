#!/bin/bash
#SBATCH --account=def-qcappart
#SBATCH --mem=32G
#SBATCH --cpus-per-task=3
#SBATCH --job-name=job_34
#SBATCH --output=knapsack/output/output_34_sg.log
#SBATCH --error=knapsack/output/error_34_sg.log
#SBATCH --time=02:33:20
module load cuda/12.2
module load python/3.10
source ../env_projet/bin/activate
python -m knapsack.run_experiments --ep 100000000 --patience 74000 --id 34 --diff IMLE --lambd 10.0 --sigma 1.0 --n_samples 1 --n 100 --dim 10 --lr 1.0 --tl 7200 --method MSE