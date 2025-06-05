#!/bin/bash
#SBATCH --time=02:00:00          # Temps d'exécution (1 heure)
#SBATCH --account=def-qcappart   # Remplacez par votre compte
#SBATCH --mem=8G                 # Mémoire requise (4 Go)
#SBATCH --cpus-per-task=4        # Nombre de CPU
# #SBATCH --gres=gpu:1             # Si vous avez besoin d'un GPU
#SBATCH --job-name=job_test6    # Nom du job
#SBATCH --output=output6.log     # Fichier de sortie
#SBATCH --error=error6.log

module load StdEnv/2023
module load cuda/12.2
module load python/3.10
module load scipy-stack
source ~/env_projet/bin/activate

for TIME_LIMIT in 60 300 600; do
  echo "Running with time_limit=$TIME_LIMIT for n=50"
  python -m portfolio.run_experiments \
    --n 50 \
    --ep_mse 1000000 \
    --method IMLE \
    --n_samples 1 --lambda_imle 10 --sigma 1.0 \
    --step_mu 0 \
    --n_iter_mu 0 \
    --out_file portfolio/results_temps_deg1.csv \
    --time_limit "$TIME_LIMIT"\
    --deg 1
done

# python -m knapsack.run_experiments --keep 1 --diff IMLE --method cla --ep 1000000 --tl 600
# python -m knapsack.run_experiments --keep 1 --diff IMLE --method LD --ep 1000000 --tl 600
# python -m knapsack.run_experiments --keep 1 --diff IMLE --method SG --ep 1000000 --tl 600 --step_mu 1 --n_iter_mu 10
# python -m knapsack.run_experiments --keep 2 --diff IMLE --method LD --ep 1000000 --tl 600
# python -m knapsack.run_experiments --keep 2 --diff IMLE --method SG --ep 1000000 --tl 600 --step_mu 1 --n_iter_mu 10

# python -m knapsack.gen_data  --n 50 --dim 10 --keep 5 --n_iter 200
