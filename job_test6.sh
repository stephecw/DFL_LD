#!/bin/bash
#SBATCH --time=13:00:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=8G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=8        # Nombre de CPU
#SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test6    # Nom du job
#SBATCH --output=output6.log     # Fichier de sortie
#SBATCH --error=error6.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ~/env_projet/bin/activate

python -m knapsack.gen_data  --n 50 --dim 10 --keep 5 --n_iter 200

# python check_mu.py --n 50 --n_train 500 --n_eval 100 --gamma 2.25 --lin 0