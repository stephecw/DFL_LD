#!/bin/bash
#SBATCH --account=def-qcappart
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --job-name=job_502
#SBATCH --output=knapsack/output/output_502_sg.log
#SBATCH --error=knapsack/output/error_502_sg.log
#SBATCH --time=00:17:40
module load cuda/12.2
module load python/3.10
source ../env_projet/bin/activate
python -m knapsack.run_experiments --ep 100000000 --patience 150 --id 502 --diff SPOPlus --n 100 --dim 10 --lr 0.01 --tl 60 --method SG --step_mu 30 --n_iter_mu 5