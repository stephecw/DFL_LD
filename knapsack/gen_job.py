l_methods = ["SG"]

l_patience = {
    "IMLE": {
        "MSE": {'30': 300, '60': 700, '300': 3000, '600': 7000, '3600': 37000, '7200': 74000},
        "cla": {'30': 2, '60': 2, '300': 3, '600': 5, '3600': 7, '7200': 15}
    },
    "SPOPlus": {
        "cla": {'30': 5, '60': 10, '300': 30, '600': 40, '3600': 60, '7200': 80}
    }
}


with open("hp_knapsack.txt") as f:
    lines = f.readlines()

diff = ""
hp_dic = {}
hp_dic['l_param'] = []
hp_dic['l_diff'] = []
for line in lines:
    if line[0]=='\n' or line[0]=='#':
        continue
    if line[0] == "/":
        diff = line[1:-1]
        hp_dic['l_diff'].append(diff)
        hp_dic[diff] = {}
        hp_dic[diff]['l_param'] = []
        continue
    elif line[0] == '*':
        diff = 'SG'
        hp_dic[diff] = {}
        hp_dic[diff]['l_param'] = []
        continue
    line = line.split(":")
    hp, type_ = line[0].split(",")
    grid = line[1].split(",")
    if type_ == "int":
        grid = list(map(int, grid)) 
    elif type_ == "float": 
        grid = list(map(float, grid)) 
    if diff == "":
        hp_dic[hp] = grid
        hp_dic['l_param'].append(hp)
    else:
        hp_dic[diff][hp] = grid
        hp_dic[diff]['l_param'].append(hp)

print(hp_dic)

jobs = []

for m in hp_dic["l_diff"]:
    mode_jobs = [{'diff':m}]
    for p in hp_dic[m]["l_param"]:
        jobs_prov = []
        for job in mode_jobs:
            for val in hp_dic[m][p]:
                new_job = job.copy()
                new_job[p] = val
                jobs_prov.append(new_job)
        mode_jobs = jobs_prov.copy()
    jobs += mode_jobs.copy()

for p in hp_dic["l_param"]:
    jobs_prov = []
    for job in jobs:
        for val in hp_dic[p]:
            new_job = job.copy()
            new_job[p] = val
            jobs_prov.append(new_job)
    jobs = jobs_prov.copy()

jobs_prov = []
for job in jobs:
    for method in l_methods:
        if method == "MSE" and job['diff'] != hp_dic['l_diff'][0]:
            continue
        new_job = job.copy()
        new_job['method'] = method
        jobs_prov.append(new_job)
jobs = jobs_prov.copy()

for p in hp_dic['SG']["l_param"]:
    jobs_prov = []
    for job in jobs:
        if job['method'] == 'SG':
            for val in hp_dic['SG'][p]:
                new_job = job.copy()
                new_job[p] = val
                jobs_prov.append(new_job)
        else:
            jobs_prov.append(job)
    jobs = jobs_prov.copy()
    
jobs_prov = []
for job in jobs:
    jobs_prov.append(job)
    if job['method'] == 'SG' and job['diff'] == 'IMLE':
        new_job = job.copy()
        new_job['loss'] = 1
        jobs_prov.append(new_job)
jobs = jobs_prov.copy()

        
print(f"Generating {len(jobs)} job files...")
offset = 54

for i, job in enumerate(jobs):
    seconds = job['tl']+2500
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    if job['method'] == 'SG':
        patience = job['step_mu'] * 5
    else:
        patience = l_patience[job['diff']][job['method']][str(job['tl'])]
    with open(f"jobfiles/jobfile{i+offset}.sh", mode="w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --account=def-qcappart\n")
        f.write(f"#SBATCH --mem={16 if job['tl']<=600 else 32}G\n")
        f.write("#SBATCH --cpus-per-task=3\n")
        f.write(f"#SBATCH --job-name=job_{i+offset}\n")
        f.write(f"#SBATCH --output=knapsack/output/output_{i+offset}_sg.log\n")
        f.write(f"#SBATCH --error=knapsack/output/error_{i+offset}_sg.log\n")
        f.write(f"#SBATCH --time={hours:02}:{minutes:02}:{seconds:02}\n")
        f.write("module load cuda/12.2\n")
        f.write("module load python/3.10\n")
        f.write("source ../env_projet/bin/activate\n")
        f.write(f"python -m knapsack.run_experiments --ep 100000000 --patience {patience} --id {i+offset}")
        for _, (p, val) in enumerate(job.items()):
            f.write(f" --{p} {val}")
