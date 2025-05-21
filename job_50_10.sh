#!/bin/bash
#SBATCH --time=08:00:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=4G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=1        # Nombre de CPU
#SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test    # Nom du job
#SBATCH --output=output_50_10.log     # Fichier de sortie
#SBATCH --error=error_50_10.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
source ../env_projet/bin/activate

python -m knapsack.run_experiments --n 50 --dim 10 --ep_cla -1 --ep_sg -1 --ep_ld -1 --ep_mse -1 --tl_cla 7200 --ep_ld 7200 --ep_mse 7200
