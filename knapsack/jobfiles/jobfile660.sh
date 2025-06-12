#!/bin/bash
#SBATCH --account=def-qcappart
#SBATCH --mem=16G
#SBATCH --cpus-per-task=3
#SBATCH --job-name=job_660
#SBATCH --output=knapsack/output/output_660_sg.log
#SBATCH --error=knapsack/output/error_660_sg.log
#SBATCH --time=00:21:40
module load cuda/12.2
module load python/3.10
source ../env_projet/bin/activate
python -m knapsack.run_experiments --ep 100000000 --patience 150 --id 660 --diff SPOPlus --n 100 --dim 10 --lr 1.0 --tl 300 --method SG --step_mu 30 --n_iter_mu 30