#!/bin/bash
#SBATCH --time=00:30:00          # Temps d'exécution
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=8G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=8        # Nombre de CPU
#SBATCH --job-name=test    # Nom du job
#SBATCH --output=knapsack/output/output.log     # Fichier de sortie
#SBATCH --error=knapsack/output/error.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
source ~/env_projet/bin/activate

python -m knapsack.gen_data --n 100 --dim 10 --n_train 50 --n_test 0 --n_eval 0 --keep 0