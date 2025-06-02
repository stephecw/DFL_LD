#!/bin/bash
#SBATCH --time=02:00:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=8G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=4        # Nombre de CPU
#SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test6    # Nom du job
#SBATCH --output=output6.log     # Fichier de sortie
#SBATCH --error=error6.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ~/env_projet/bin/activate

python -m knapsack.run_experiments --keep 1 --diff IMLE --method cla --ep 1000000 --tl 600
python -m knapsack.run_experiments --keep 1 --diff IMLE --method LD --ep 1000000 --tl 600
python -m knapsack.run_experiments --keep 1 --diff IMLE --method SG --ep 1000000 --tl 600 --step_mu 1 --n_iter_mu 10
python -m knapsack.run_experiments --keep 2 --diff IMLE --method LD --ep 1000000 --tl 600
python -m knapsack.run_experiments --keep 2 --diff IMLE --method SG --ep 1000000 --tl 600 --step_mu 1 --n_iter_mu 10

# python -m knapsack.gen_data  --n 50 --dim 10 --keep 5 --n_iter 200
