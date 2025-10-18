#!/usr/bin/env python3

import os

# Hyperparameter grid
grids = {
    'lr': [0.0001, 0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 1],
}

# Constants for all jobs
time = "02:00:00"
account = "def-qcappart"
cpus = 1
mem = "8G"
n = 50
ep_cla = 1000000
method = "SPOPlus"  # Choix du méthode
seed = 0
report = "10 60 300 600 1800 3600"

# Directories
script_dir = "."  # Génère les scripts ici
log_dir = "logs"
out_dir = "portfolio/n50"

# SLURM script template
template = """#!/bin/bash
#SBATCH --time={time}
#SBATCH --account={account}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --job-name=portfolio_cla_{method}_n{n}_lr{lr_str}_seed{seed}
#SBATCH --output={log_dir}/portfolio_cla_{method}_n{n}_lr{lr_str}_seed{seed}.out
#SBATCH --error={log_dir}/portfolio_cla_{method}_n{n}_lr{lr_str}_seed{seed}.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \
  --n {n} \
  --ep_cla {ep_cla} --lr {lr} \
  --method {method} \
  --seed {seed} \
  --report {report} \
  --out_file {out_dir}/results_hyper_seed{seed}.csv \
  --time_limit 7200
"""

def main():
    for lr in grids['lr']:
        lr_str = str(lr).replace('.', 'p')
        script_name = os.path.join(script_dir, f"job_cla_spoplus_lr{lr_str}.sh")
        with open(script_name, 'w') as f:
            f.write(
                template.format(
                    time=time,
                    account=account,
                    cpus=cpus,
                    mem=mem,
                    method=method,
                    n=n,
                    ep_cla=ep_cla,
                    lr=lr,
                    lr_str=lr_str,
                    seed=seed,
                    report=report,
                    log_dir=log_dir,
                    out_dir=out_dir,
                )
            )
        os.chmod(script_name, 0o755)
        print(f"Generated {script_name}")

if __name__ == '__main__':
    main()