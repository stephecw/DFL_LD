#!/bin/bash
#SBATCH --time=04:00:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=4G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=4        # Nombre de CPU
# #SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test1    # Nom du job
#SBATCH --output=output1.log     # Fichier de sortie
#SBATCH --error=error1.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.gen_data --n 100 --n_iter 1000 --deg 1

python -m portfolio.gen_data --n 100 --n_iter 1000 --deg 1 --n_train 1000 --n_validation 250

# python -m knapsack.run_experiments --keep 2 --diff IMLE --method LD --ep 1000000 --tl 10800

# for TIME_LIMIT in 1800; do
#   echo "Running with time_limit=$TIME_LIMIT for n=30"
#   python -m portfolio.run_experiments \
#     --n 100 \
#     --ep_mse 1000000 \
#     --method IMLE \
#     --n_samples 1 --lambda_imle 10 --sigma 1.0 \
#     --step_mu 0 \
#     --n_iter_mu 0 \
#     --out_file portfolio/results_temps.csv \
#     --time_limit "$TIME_LIMIT"

  # echo "Running with time_limit=$TIME_LIMIT for n=50"
  # python -m portfolio.run_experiments \
  #   --n 100 \
  #   --ep_cla 1000000 \
  #   --method SPOPlus \
  #   --n_samples 5 --lambda_imle 10 --sigma 1.0 \
  #   --step_mu 0 \
  #   --n_iter_mu 0 \
  #   --out_file portfolio/results_temps.csv \
  #   --time_limit "$TIME_LIMIT"
# done
