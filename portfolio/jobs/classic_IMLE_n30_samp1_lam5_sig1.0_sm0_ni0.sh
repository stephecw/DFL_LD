#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --account=def-qcappart
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=classic_IMLE_n30_samp1_lam5_sig1.0_sm0_ni0
#SBATCH --output=logs/classic_IMLE_n30_samp1_lam5_sig1.0_sm0_ni0.out
#SBATCH --error=logs/classic_IMLE_n30_samp1_lam5_sig1.0_sm0_ni0.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 30 \
  --ep_cla 10000 \
  --method IMLE \
  --n_samples 1 --lambda_imle 5 --sigma 1.0 \
  --step_mu 0 \
  --n_iter_mu 0 \
  --out_file results.csv
