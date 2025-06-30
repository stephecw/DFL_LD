import os

def modifier_fichier_sh(chemin_fichier):
    with open(chemin_fichier, 'r') as fichier:
        lignes = fichier.readlines()

    lignes_modifiees = []
    for ligne in lignes:
        if ligne.startswith('#SBATCH --cpus-per-task='):
            ligne = '#SBATCH --cpus-per-task=1\n'
        lignes_modifiees.append(ligne)

    with open(chemin_fichier, 'w') as fichier:
        fichier.writelines(lignes_modifiees)
    print(f"Modifié : {chemin_fichier}")

def modifier_tous_les_fichiers_sh(dossier):
    for fichier in os.listdir(dossier):
        if fichier.endswith('.sh'):
            chemin_complet = os.path.join(dossier, fichier)
            modifier_fichier_sh(chemin_complet)

if __name__ == '__main__':
    dossier_courant = os.path.dirname(os.path.abspath(__file__))
    modifier_tous_les_fichiers_sh(dossier_courant)