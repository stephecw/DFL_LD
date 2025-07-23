#!/usr/bin/env python3

import os
import re

# Motif strict : fichier commençant par job_sg_lr ou job_ld_lr
pattern = re.compile(r'^job_(sg|ld)_lr.*\.sh$')

current_dir = os.getcwd()
found = False

for filename in os.listdir(current_dir):
    if not pattern.match(filename):
        continue

    found = True
    print(f"🔍 Traitement de : {filename}")

    with open(filename, 'r') as f:
        lines = f.readlines()

    new_lines = []
    inserted = False

    for line in lines:
        if 'python -m portfolio.run_experiments' in line and '--muloss' not in line:
            line = line.rstrip() + '   --muloss 0\n'
            inserted = True
        new_lines.append(line)

    if inserted:
        new_filename = filename.replace('.sh', '_muloss.sh')
        with open(new_filename, 'w') as f:
            f.writelines(new_lines)
        os.chmod(new_filename, 0o755)
        print(f"✅ Fichier créé : {new_filename}")
    else:
        print(f"⚠️  Aucun ajout (--muloss déjà présent ?) dans : {filename}")

if not found:
    print("❌ Aucun fichier job_sg_lr*.sh ou job_ld_lr*.sh trouvé.")