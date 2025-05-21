#!/bin/bash
#SBATCH --time=03:00:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=4G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=1        # Nombre de CPU
#SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test4    # Nom du job
#SBATCH --output=output.log     # Fichier de sortie
#SBATCH --error=error.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ~/env_projet/bin/activate

python -m knapsack.gen_data --n_train 0 --n_test 200 --n 30 --dim 10
