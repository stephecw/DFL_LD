import itertools
import os

# 1) Dimensions et jobtypes
ns = [30]#, 50, 100]
jobtypes = {
    'classic': {'ep': '--ep_cla', 'epochs': 10000},
    'LD'     : {'ep': '--ep_ld',  'epochs': 10000},
    'SG'     : {'ep': '--ep_sg',  'epochs': 10000}
}

# 2) Méthodes à tester, par jobtype
methods_for_job = {
    'classic': ['IMLE'],#['IMLE', 'SPOPlus'],            # plus de 'Exact' ici !
    'LD'     : ['IMLE'],#['IMLE', 'SPOPlus', 'Exact'],
    'SG'     : ['IMLE']#['IMLE', 'SPOPlus', 'Exact']
}

# 3) Hyper‑paramètres
n_samples_list = [1, 5, 10]
lambda_list    = [1, 5, 10, 20]
sigma_list     = [1.0]
step_mu_list   = [5, 10, 15, 20]
n_iter_mu_list = [10, 20, 30]

# 4) Template SLURM
template = """#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --account=def-qcappart
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --job-name={jobname}
#SBATCH --output=logs/{jobname}.out
#SBATCH --error=logs/{jobname}.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \\
  --n {n} \\
  {ep_flag} {epochs} \\
  --method {method} \\
  {n_samples_flag} {lambda_flag} {sigma_flag} \\
  --step_mu {step_mu} \\
  --n_iter_mu {n_iter_mu} \\
  --out_file results.csv
"""

os.makedirs('jobs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# 5) Génération
for n in ns:
    for jobtype, cfg in jobtypes.items():
        for method in methods_for_job[jobtype]:
            # grille IMLE uniquement si méthode IMLE
            imle_grid = (
                itertools.product(n_samples_list, lambda_list, sigma_list)
                if method == 'IMLE' else
                [(None, None, None)]
            )
            for n_samples, lambd, sigma in imle_grid:
                # pour SG on balaie aussi step_mu/n_iter_mu
                sweeps = (
                    itertools.product(step_mu_list, n_iter_mu_list)
                    if jobtype == 'SG' else
                    [(0, 0)]
                )
                for step_mu, n_iter_mu in sweeps:
                    jobname = f"{jobtype}_{method}_n{n}_samp{n_samples}_lam{lambd}_sig{sigma}_sm{step_mu}_ni{n_iter_mu}"
                    flags = {
                        'jobname': jobname,
                        'n': n,
                        'ep_flag': cfg['ep'],
                        'epochs': cfg['epochs'],
                        'method': method,
                        'n_samples_flag': f"--n_samples {n_samples}" if n_samples else "",
                        'lambda_flag':    f"--lambda_imle {lambd}"  if lambd else "",
                        'sigma_flag':     f"--sigma {sigma}"        if sigma else "",
                        'step_mu':        step_mu,
                        'n_iter_mu':      n_iter_mu
                    }
                    content = template.format(**flags)
                    with open(f"jobs/{jobname}.sh", 'w') as f:
                        f.write(content)