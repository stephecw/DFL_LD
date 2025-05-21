#!/bin/bash
#SBATCH --time=01:00:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=4G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=1        # Nombre de CPU
#SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test50    # Nom du job
#SBATCH --output=output50.log     # Fichier de sortie
#SBATCH --error=error50.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments --n 50 --ep_cla 400 --ep_sg 200 --ep_mse 400
