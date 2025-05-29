#!/bin/bash
#SBATCH --time=04:30:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=4G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=1        # Nombre de CPU
#SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test3    # Nom du job
#SBATCH --output=output3.log     # Fichier de sortie
#SBATCH --error=error3.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ~/env_projet/bin/activate

for TIME_LIMIT in 60 300 600 1800; do
  echo "Running with time_limit=$TIME_LIMIT for n=30"
  python -m portfolio.run_experiments \
    --n 30 \
    --ep_ld 1000000 \
    --method IMLE \
    --n_samples 1 --lambda_imle 10 --sigma 1.0 \
    --step_mu 5 \
    --n_iter_mu 30 \
    --out_file portfolio/results_temps.csv \
    --time_limit "$TIME_LIMIT"

  echo "Running with time_limit=$TIME_LIMIT for n=50"
  python -m portfolio.run_experiments \
    --n 50 \
    --ep_ld 1000000 \
    --method IMLE \
    --n_samples 10 --lambda_imle 1 --sigma 1.0 \
    --step_mu 15 \
    --n_iter_mu 10 \
    --out_file portfolio/results_temps.csv \
    --time_limit "$TIME_LIMIT"
done