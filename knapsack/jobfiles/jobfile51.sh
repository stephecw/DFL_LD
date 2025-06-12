#!/bin/bash
#SBATCH --account=def-qcappart
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --job-name=job_51
#SBATCH --output=knapsack/output/output_51_sg.log
#SBATCH --error=knapsack/output/error_51_sg.log
#SBATCH --time=00:26:40
module load cuda/12.2
module load python/3.10
source ../env_projet/bin/activate
python -m knapsack.run_experiments --ep 100000000 --patience 40 --id 51 --diff SPOPlus --n 100 --dim 10 --lr 1.0 --tl 600 --method cla