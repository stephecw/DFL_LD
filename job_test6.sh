#!/bin/bash
#SBATCH --time=06:00:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=4G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=4        # Nombre de CPU
#SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test6    # Nom du job
#SBATCH --output=output6.log     # Fichier de sortie
#SBATCH --error=error6.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.gen_data --n 30 --n_iter 2000
python -m portfolio.gen_data --n 50 --n_iter 2000
python -m portfolio.gen_data --n 100 --n_iter 2000
python -m portfolio.run_experiments --n 50 --ep_cla 0 --ep_sg 100 --ep_mse 0 --ep_ld 0 --method IMLE 
python -m portfolio.run_experiments --n 50 --ep_cla 0 --ep_sg 0 --ep_mse 0 --ep_ld 100 --method IMLE 

# python check_mu.py --n 50 --n_train 500 --n_eval 100 --gamma 2.25 --lin 0