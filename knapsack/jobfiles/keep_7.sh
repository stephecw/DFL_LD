#!/bin/bash
#SBATCH --time=03:00:00          # Temps d'exécution
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=4G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=8        # Nombre de CPU
#SBATCH --job-name=data7    # Nom du job
#SBATCH --output=knapsack/output/output7.log     # Fichier de sortie
#SBATCH --error=knapsack/output/error7.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
source ~/env_projet/bin/activate

python -m knapsack.gen_data --keep 7 --n 100 --dim 10 --n_iter 1000