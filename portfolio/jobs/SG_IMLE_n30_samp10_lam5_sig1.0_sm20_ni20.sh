#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --account=def-qcappart
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=SG_IMLE_n30_samp10_lam5_sig1.0_sm20_ni20
#SBATCH --output=logs/SG_IMLE_n30_samp10_lam5_sig1.0_sm20_ni20.out
#SBATCH --error=logs/SG_IMLE_n30_samp10_lam5_sig1.0_sm20_ni20.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n 30 \
  --ep_sg 10000 \
  --method IMLE \
  --n_samples 10 --lambda_imle 5 --sigma 1.0 \
  --step_mu 20 \
  --n_iter_mu 20 \
  --out_file results.csv
