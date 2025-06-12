#!/bin/bash
#SBATCH --time=02:00:00          # Temps d'exécution
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=8G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=2        # Nombre de CPU
#SBATCH --job-name=data_test    # Nom du job
#SBATCH --output=knapsack/output/output_test.log     # Fichier de sortie
#SBATCH --error=knapsack/output/error_test.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
source ~/env_projet/bin/activate

python -m knapsack.gen_data --keep 0 --n 100 --dim 10 --n_train 200 --n_test 1000 --n_eval 100 --deg 8 --n_iter 1000000