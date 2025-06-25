import itertools
import os

ns = [100]
methods = ['IMLE', 'SPOPlus']
step_mu_list = [1, 5, 10, 20]
n_iter_mu_list = [5,10, 20]
report_times = [60, 600, 1800, 3600]

n_samples_list = [1]
lambda_list = [10]
sigma_list = [1]

template = """#!/bin/bash
#SBATCH --time=04:30:00
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
  --method {job} \\
  --n {n} \\
  --ep {ep} \\
  --tl {tl} \\
  --step_mu {step_mu} \\
  --n_iter_mu {n_iter_mu} \\
  --report {report_str} \\
  --out_file {out_file} \\
  --seed 0 \\
{extra_args}
"""

# Créer les dossiers si nécessaire
os.makedirs('knapsack/jobs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# for n in ns:
#     for method in methods:
#         for step_mu, n_iter_mu in itertools.product(step_mu_list, n_iter_mu_list):
#             jobname = f"knapsack_SG_{method}_n{n}_sm{step_mu}_ni{n_iter_mu}"
#             report_str = ' '.join(str(t) for t in report_times)

#             extra_args = ""

#             flags = {
#                 'jobname': jobname,
#                 'job': "SG",
#                 'method': method,
#                 'n': n,
#                 'ep': 1000000,
#                 'tl': 3600,
#                 'step_mu': step_mu,
#                 'n_iter_mu': n_iter_mu,
#                 'report_str': report_str,
#                 'out_file': "knapsack/seed/results_seed0.csv",
#                 'extra_args': extra_args
#             }

#             content = template.format(**flags)

#             with open(f"knapsack/jobs/{jobname}.sh", 'w') as f:
#                 f.write(content)

os.makedirs('knapsack/jobs_mains_sum', exist_ok=True)
os.makedirs('logs', exist_ok=True)

for n in ns:
    for method in methods:
        for step_mu, n_iter_mu in itertools.product(step_mu_list, n_iter_mu_list):
            jobname = f"knapsack_SG_{method}_n{n}_sm{step_mu}_ni{n_iter_mu}_mains"
            report_str = ' '.join(str(t) for t in report_times)

            extra_args = "--mains 0 1 --combine sum"

            flags = {
                'jobname': jobname,
                'job': "SG",
                'method': method,
                'n': n,
                'ep': 1000000,
                'tl': 3600,
                'step_mu': step_mu,
                'n_iter_mu': n_iter_mu,
                'report_str': report_str,
                'out_file': "knapsack/seed/results_seed0_mains_sum.csv",
                'extra_args': extra_args
            }

            content = template.format(**flags)

            with open(f"knapsack/jobs_mains_sum/{jobname}.sh", 'w') as f:
                f.write(content)

os.makedirs('knapsack/jobs_muloss', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# for n in ns:
#     for method in ['IMLE']:
#         for step_mu, n_iter_mu in itertools.product(step_mu_list, n_iter_mu_list):
#             jobname = f"knapsack_SG_{method}_n{n}_sm{step_mu}_ni{n_iter_mu}_muloss"
#             report_str = ' '.join(str(t) for t in report_times)

#             extra_args = "--muloss 0"

#             flags = {
#                 'jobname': jobname,
#                 'job': "SG",
#                 'method': method,
#                 'n': n,
#                 'ep': 1000000,
#                 'tl': 3600,
#                 'step_mu': step_mu,
#                 'n_iter_mu': n_iter_mu,
#                 'report_str': report_str,
#                 'out_file': "knapsack/seed/results_seed0_muloss.csv",
#                 'extra_args': extra_args
#             }

#             content = template.format(**flags)

#             with open(f"knapsack/jobs_muloss/{jobname}.sh", 'w') as f:
#                 f.write(content)

os.makedirs('knapsack/jobs_muloss_mains_sum', exist_ok=True)
os.makedirs('logs', exist_ok=True)

for n in ns:
    for method in ['IMLE']:
        for step_mu, n_iter_mu in itertools.product(step_mu_list, n_iter_mu_list):
            jobname = f"knapsack_SG_{method}_n{n}_sm{step_mu}_ni{n_iter_mu}_muloss_mains"
            report_str = ' '.join(str(t) for t in report_times)

            extra_args = "--muloss 0 --mains 0 1 --combine sum"

            flags = {
                'jobname': jobname,
                'job': "SG",
                'method': method,
                'n': n,
                'ep': 1000000,
                'tl': 3600,
                'step_mu': step_mu,
                'n_iter_mu': n_iter_mu,
                'report_str': report_str,
                'out_file': "knapsack/seed/results_seed0_muloss_mains_sum.csv",
                'extra_args': extra_args
            }

            content = template.format(**flags)

            with open(f"knapsack/jobs_muloss_mains_sum/{jobname}.sh", 'w') as f:
                f.write(content)

os.makedirs('knapsack/jobs', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# for n in ns:
#     for method in methods:
#         for step_mu, n_iter_mu in itertools.product(step_mu_list, n_iter_mu_list):
#             jobname = f"knapsack_cla_{method}_n{n}_sm{step_mu}_ni{n_iter_mu}"
#             report_str = ' '.join(str(t) for t in report_times)

#             extra_args = ""

#             flags = {
#                 'jobname': jobname,
#                 'job': "cla",
#                 'method': method,
#                 'n': n,
#                 'ep': 1000000,
#                 'tl': 3600,
#                 'step_mu': step_mu,
#                 'n_iter_mu': n_iter_mu,
#                 'report_str': report_str,
#                 'out_file': "knapsack/seed/results_seed0.csv",
#                 'extra_args': extra_args
#             }

#             content = template.format(**flags)

#             with open(f"knapsack/jobs/{jobname}.sh", 'w') as f:
#                 f.write(content)

# for n in ns:
#     for method in ["SPOPlus"]:
#         for step_mu, n_iter_mu in itertools.product(step_mu_list, n_iter_mu_list):
#             jobname = f"knapsack_MSE_{method}_n{n}_sm{step_mu}_ni{n_iter_mu}"
#             report_str = ' '.join(str(t) for t in report_times)

#             extra_args = ""

#             flags = {
#                 'jobname': jobname,
#                 'job': "MSE",
#                 'method': method,
#                 'n': n,
#                 'ep': 1000000,
#                 'tl': 3600,
#                 'step_mu': step_mu,
#                 'n_iter_mu': n_iter_mu,
#                 'report_str': report_str,
#                 'out_file': "knapsack/seed/results_seed0.csv",
#                 'extra_args': extra_args
#             }

#             content = template.format(**flags)

#             with open(f"knapsack/jobs/{jobname}.sh", 'w') as f:
#                 f.write(content)
