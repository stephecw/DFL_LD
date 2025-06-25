import os
import itertools

# Grille de paramètres
step_mu_list = [20]
n_iter_mu_list = [10, 20]
seeds = list(range(10))

# Paramètres fixes
n = 50
ep = 1000000
report_times = [10, 60, 300, 600]

# SLURM script template sans optional_flags
template = """#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name={jobname}
#SBATCH --output=logs/{jobname}.out
#SBATCH --error=logs/{jobname}.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \\
  --n {n} \\
  {train_flag} \\
  --method {method} \\
  --seed {seed} \\
  --report {report_str} \\
  --out_file {out_file} \\
  --time_limit 600 \\
  --muloss 0
"""

# Création des répertoires
os.makedirs("portfolio/jobs_muloss", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("portfolio/seed", exist_ok=True)

# Boucle SG + Exact
for step_mu, n_iter_mu, seed in itertools.product(step_mu_list, n_iter_mu_list, seeds):
    jobname = f"portfolio_SG_Exact_n{n}_sm{step_mu}_ni{n_iter_mu}_seed{seed}"
    report_str = ' '.join(str(t) for t in report_times)
    out_file = f"portfolio/seed/results_seed{seed}_muloss.csv"

    script_content = template.format(
        jobname=jobname,
        n=n,
        train_flag=f"--ep_sg {ep} --step_mu {step_mu} --n_iter_mu {n_iter_mu}",
        method="Exact",
        seed=seed,
        report_str=report_str,
        out_file=out_file
    )

    with open(f"portfolio/jobs_muloss/{jobname}.sh", "w") as f:
        f.write(script_content)

# Boucle LD + Exact 
for seed in seeds:
    jobname = f"portfolio_LD_Exact_n{n}_seed{seed}"
    report_str = ' '.join(str(t) for t in report_times)
    out_file = f"portfolio/seed/results_seed{seed}_muloss.csv"

    script_content = template.format(
        jobname=jobname,
        n=n,
        train_flag=f"--ep_ld {ep}",
        method="Exact",
        seed=seed,
        report_str=report_str,
        out_file=out_file
    )

    with open(f"portfolio/jobs_muloss/{jobname}.sh", "w") as f:
        f.write(script_content)

# # Boucle classic + SPOPlus / IMLE, mse + SPOPlus (sans optional_flags)
# for seed in seeds:
#     report_str = ' '.join(str(t) for t in report_times)
#     out_file = f"portfolio/seed/results_seed{seed}.csv"

#     jobs = [
#         {
#             "jobname": f"portfolio_classic_SPOPlus_n{n}_seed{seed}",
#             "train_flag": f"--ep_cla {ep}",
#             "method": "SPOPlus",
#         },
#         {
#             "jobname": f"portfolio_classic_IMLE_n{n}_seed{seed}",
#             "train_flag": f"--ep_cla {ep} --lr 0.01",
#             "method": "IMLE",
#         },
#         {
#             "jobname": f"portfolio_mse_SPOPlus_n{n}_seed{seed}",
#             "train_flag": f"--ep_mse {ep}",
#             "method": "SPOPlus",
#         }
#     ]

#     for job in jobs:
#         content = template.format(
#             jobname=job["jobname"],
#             n=n,
#             train_flag=job["train_flag"],
#             method=job["method"],
#             seed=seed,
#             report_str=report_str,
#             out_file=out_file
#         )

#         with open(f"portfolio/jobs/{job['jobname']}.sh", "w") as f:
#             f.write(content)