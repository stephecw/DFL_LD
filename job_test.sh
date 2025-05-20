#!/bin/bash
#SBATCH --time=01:00:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=4G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=1        # Nombre de CPU
#SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test    # Nom du job
#SBATCH --output=output.log     # Fichier de sortie
#SBATCH --error=error.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ../env_projet/bin/activate

python -m portfolio.run_experiments --n 50 --ep_cla 5 --ep_sg 5 --ep_mse 5 --step_mu 2 --n_iter_mu 5 --method IMLE
