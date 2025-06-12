#!/bin/bash
#SBATCH --account=def-qcappart
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --job-name=job_38
#SBATCH --output=knapsack/output/output_38_sg.log
#SBATCH --error=knapsack/output/error_38_sg.log
#SBATCH --time=00:21:40
module load cuda/12.2
module load python/3.10
source ../env_projet/bin/activate
python -m knapsack.run_experiments --ep 100000000 --patience 30 --id 38 --diff SPOPlus --n 100 --dim 10 --lr 0.01 --tl 300 --method cla