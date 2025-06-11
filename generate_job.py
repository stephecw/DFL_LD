import itertools
import os

ns = [100]
methods = ['IMLE', 'SPOPlus']
step_mu_list = [1, 5, 10, 20]
n_iter_mu_list = [10, 20]
report_times = [60, 600, 1800, 3600]

n_samples_list = [10]
lambda_list = [0.1]
sigma_list = [0.1]

template = """#!/bin/bash
#SBATCH --time=01:30:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name={jobname}
#SBATCH --output=logs/{jobname}.out
#SBATCH --error=logs/{jobname}.err

module load StdEnv/2023 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m knapsack.run_experiments \\
  --diff {method} \\
  --method SG \\
  --n {n} \\
  --ep {ep} \\
  --tl {tl} \\
  --step_mu {step_mu} \\
  --n_iter_mu {n_iter_mu} \\
  --report {report_str} \\
  --out_file {out_file} \\
{extra_args}
"""

# Créer les dossiers si nécessaire
os.makedirs('knapsack/jobs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

for n in ns:
    for method in methods:
        imle_grid = (
            itertools.product(n_samples_list, lambda_list, sigma_list)
            if method == 'IMLE' else
            [(None, None, None)]
        )
        for n_samples, lambd, sigma in imle_grid:
            for step_mu, n_iter_mu in itertools.product(step_mu_list, n_iter_mu_list):
                jobname = f"knapsack_SG_{method}_n{n}_sm{step_mu}_ni{n_iter_mu}"
                report_str = ' '.join(str(t) for t in report_times)

                extra_args = ""
                if method == "IMLE":
                    extra_args = f"  --n_samples {n_samples} \\\n  --lambd {lambd} \\\n  --sigma {sigma} \\"

                flags = {
                    'jobname': jobname,
                    'method': method,
                    'n': n,
                    'ep': 1000000,
                    'tl': 3600,
                    'step_mu': step_mu,
                    'n_iter_mu': n_iter_mu,
                    'report_str': report_str,
                    'out_file': "knapsack/results_deg4.csv",
                    'extra_args': extra_args
                }

                content = template.format(**flags)

                with open(f"knapsack/jobs/{jobname}.sh", 'w') as f:
                    f.write(content)