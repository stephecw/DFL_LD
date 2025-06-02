#!/bin/bash
#SBATCH --time=07:00:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=8G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=8        # Nombre de CPU
#SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test5    # Nom du job
#SBATCH --output=output5.log     # Fichier de sortie
#SBATCH --error=error5.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ~/env_projet/bin/activate

python -m knapsack.gen_data  --n 50 --dim 10 --keep 3 --n_iter 200

# python -m knapsack.run_experiments --keep 1 --diff IMLE --method LD --ep 1000000 --tl 600
