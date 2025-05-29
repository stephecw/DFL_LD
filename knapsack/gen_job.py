l_methods = ["SG"] #["cla", "LD", "SG", "MSE"]
timetable = {
            '30,5':'04:00:00',
            '30,10':'05:00:00',
            '50,5':'02:00:00',
            '50,10':'06:00:00',
            '100,5':'06:00:00',
            '100,10':'0:00:00'
            }
timetable_sec = {
            '30,5':10800,
            '30,10':14400,
            '50,5':5000,
            '50,10':18000,
            '100,5':18000,
            '100,10':21600
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
        diff = 'MSE'
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

for p in hp_dic['MSE']["l_param"]:
    jobs_prov = []
    for job in jobs:
        if job['method'] == 'SG':
            for val in hp_dic['MSE'][p]:
                new_job = job.copy()
                new_job[p] = val
                jobs_prov.append(new_job)
        else:
            jobs_prov.append(job)
    jobs = jobs_prov.copy()
        
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
        f.write(f"#SBATCH --output=knapsack/output/output_{i}_spo.log\n")
        f.write(f"#SBATCH --error=knapsack/output/error_{i}_spo.log\n")
        f.write(f"#SBATCH --time={time}\n")
        f.write("module load cuda/12.2\n")
        f.write("module load python/3.10\n")
        f.write("source ../env_projet/bin/activate\n")
        f.write(f"python -m knapsack.run_experiments --ep 100000000 --tl {time_sec}")
        for i, (p, val) in enumerate(job.items()):
            f.write(f" --{p} {val}")
        
        
        
    


    
    