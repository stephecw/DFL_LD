timetable = {
            '30,5':'03:00:00',
            '30,10':'04:00:00',
            '50,5':'04:00:00',
            '50,10':'05:00:00',
            '100,5':'05:00:00',
            '100,10':'06:00:00'
            }
timetable_sec = {
            '30,5':9000,
            '30,10':12600,
            '50,5':12600,
            '50,10':16200,
            '100,5':16200,
            '100,10':19800
            }

with open("best_spo.txt") as f:
    lines = f.readlines()
'''    
jobs = []
n = 0
for line in lines:
    job = {}
    line = line.split(";")
    new_n = int(line[4])
    if new_n == n:
        continue
    n = new_n
    job["method"] = line[0]
    job["diff"] = line[1]
    job["dim"] = int(line[2])
    job["n"] = new_n
    job["lr"] = float(line[6])
    job["step_mu"] = int(line[7])
    job["n_iter_mu"] = int(line[8])
    jobs.append(job)
'''

jobs = []
n = 0
for line in lines[1:]:
    job = {}
    line = line.split(";")
    job["method"] = line[0]
    job["diff"] = line[1]
    job["dim"] = int(line[2])
    job["n"] = int(line[4])
    job["lr"] = float(line[6])
    job["step_mu"] = int(line[7])
    job["n_iter_mu"] = int(line[8])
    jobs.append(job)
    
print(f"Generating {len(jobs)} job files...")
    
for i, job in enumerate(jobs):
    time = timetable[f'{job['n']},{job['dim']}']
    time_sec = timetable_sec[f'{job['n']},{job['dim']}']
    with open(f"jobfiles/jobfile{i}.sh", mode="w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --account=def-qcappart\n")
        f.write("#SBATCH --mem=16G\n")
        f.write("#SBATCH --cpus-per-task=1\n")
        f.write("#SBATCH --gres=gpu:1\n")
        f.write(f"#SBATCH --job-name=job_{i}\n")
        f.write(f"#SBATCH --output=knapsack/output/output_{i}.log\n")
        f.write(f"#SBATCH --error=knapsack/output/error_{i}.log\n")
        f.write(f"#SBATCH --time={time}\n")
        f.write("module load cuda/12.2\n")
        f.write("module load python/3.10\n")
        f.write("source ../env_projet/bin/activate\n")
        f.write(f"python -m knapsack.run_experiments --ep 100000000 --tl {time_sec}")
        for i, (p, val) in enumerate(job.items()):
            f.write(f" --{p} {val}")