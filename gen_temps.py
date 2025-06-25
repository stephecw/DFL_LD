import os

learning_rates = [0.01, 0.002, 0.001, 0.0005]
job_dir = "portfolio/jobs_lr"
os.makedirs(job_dir, exist_ok=True)

for lr in learning_rates:
    lr_str = str(lr).replace('.', 'p')
    filename = os.path.join(job_dir, f"job_sg_exact_2_lr_{lr_str}.sh")

    with open(filename, "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --account=def-qcappart
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --job-name=portfolio_SG_Exact_lr{lr_str}
#SBATCH --output=logs/portfolio_SG_Exact_lr{lr_str}.out
#SBATCH --error=logs/portfolio_SG_Exact_lr{lr_str}.err

module load StdEnv/2023 cuda/12.2 python/3.10 scipy-stack
source ~/env_projet/bin/activate

python -m portfolio.run_experiments \\
  --n 50 \\
  --ep_sg 1000000 --step_mu 20 --n_iter_mu 20 \\
  --method Exact \\
  --seed 0 \\
  --report 10 60 300 600 \\
  --out_file portfolio/seed/results_lr.csv \\
  --time_limit 600 \\
  --lr {lr}
""")

    print(f"✅ Job créé : {filename}")