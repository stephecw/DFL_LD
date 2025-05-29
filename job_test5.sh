#!/bin/bash
#SBATCH --time=01:20:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=4G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=1        # Nombre de CPU
#SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test5    # Nom du job
#SBATCH --output=output5.log     # Fichier de sortie
#SBATCH --error=error5.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments --n 50 --ep_cla 0 --ep_sg 500 --ep_mse 0 --ep_ld 0 --method Exact
