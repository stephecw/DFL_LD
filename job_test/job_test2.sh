#!/bin/bash
#SBATCH --time=02:00:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=4G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=1        # Nombre de CPU
#SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test2    # Nom du job
#SBATCH --output=output2.log     # Fichier de sortie
#SBATCH --error=error2.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ~/env_projet/bin/activate

python -m knapsack.run_experiments --keep 1 --diff IMLE --method LD --ep 1000000 --tl 600

# for TIME_LIMIT in 300 1800; do
#   echo "Running with time_limit=$TIME_LIMIT for n=30"
#   python -m portfolio.run_experiments \
#     --n 100 \
#     --ep_sg 1000000 \
#     --method Exact \
#     --n_samples 1 --lambda_imle 10 --sigma 1.0 \
#     --step_mu 1 \
#     --n_iter_mu 10 \
#     --out_file portfolio/results_temps.csv \
#     --time_limit "$TIME_LIMIT"

#   echo "Running with time_limit=$TIME_LIMIT for n=50"
#   python -m portfolio.run_experiments \
#     --n 100 \
#     --ep_ld 1000000 \
#     --method Exact \
#     --n_samples 1 --lambda_imle 10 --sigma 1.0 \
#     --step_mu 1 \
#     --n_iter_mu 10 \
#     --out_file portfolio/results_temps.csv \
#     --time_limit "$TIME_LIMIT"
# done