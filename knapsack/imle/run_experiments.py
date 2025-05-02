import sched
import torch
from torch import optim
from data_import import ImportDataset
from imle.train_imle import train, train_LD, train_SG

from imle.train_imle import CustomMLP

import argparse

# Définir les arguments de ligne de commande
parser = argparse.ArgumentParser(description="Script d'entraînement avec des dimensions spécifiées.")
parser.add_argument('--dim', type=int, default=5, help='Nombre de contraintes.')
parser.add_argument('--n', type=int, default=30, help='Nombre d\'item.')
parser.add_argument('--ep_cla', type=int, default=0, help='Nombre d\'epochs pour l\'entraînement classique. (0 pour ne pas l\'exécuter)')
parser.add_argument('--ep_ld', type=int, default=0, help='Nombre d\'epochs pour l\'entraînement LD. (0 pour ne pas l\'exécuter)')
parser.add_argument('--ep_sg', type=int, default=0, help='Nombre d\'epochs pour l\'entraînement SG. (0 pour ne pas l\'exécuter)')
parser.add_argument('--step_mu', type=int, default=5, help='Nombre d\'epochs entre la mise à jour des \mu. (0 pour ne pas l\'exécuter)')
parser.add_argument('--n_iter_mu', type=int, default=10, help='Nombre d\'itérations pour l\'optimisation de \mu. (0 pour ne pas l\'exécuter)')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Entraînement sur :", device)

def run_train(model, jobtype, dim, num_feat, num_item, num_data_train, num_data_test, 
              batch_size=32, epochs=20, lr=1e-3, 
              schedulerType="StepLR", sched_step_size=50, sched_gamma=0.5,
              IMLE_n_samples=10, IMLE_sigma=1.0, IMLE_lambd=10, IMLE_two_sides=False, IMLE_processes=1,
              verbose=False, wandbarg=None, save_model=True, step_mu=5, n_iter_mu=15):
    """
    Fonction principale pour charger le dataset et entraîner le modèle.
    model : nn.Module : Modèle à entraîner.
    LD : bool : Si True, utilise la décomposition lagrangienne.
    dim : int : Nombre de dimensions.
    num_feat : int : Nombre de features.
    num_item : int : Nombre d'items.
    num_data_train : int : Nombre de données d'entraînement.
    batch_size : int : Taille des batch.
    epochs : int : Nombre d'époques d'entraînement.
    lr : float : Taux d'apprentissage.
    schedulerType : str : Type de scheduler à utiliser. 'StepLR' ou 'None'.
    sched_step_size : int : Step size pour le scheduler.
    sched_gamma : float : Gamma pour le scheduler.
    IMLE_n_samples : int : Nombre d'échantillons Monte Carlo pour l'IMLE.
    IMLE_sigma : float : Sigma pour l'IMLE.
    IMLE_lambd : float : Lambda pour l'IMLE.
    IMLE_two_sides : bool : Si True, utilise une perturbation à deux côtés.
    IMLE_processes : int : Nombre de processus pour l'IMLE.
    verbose : bool : Si True, affiche l'avancement.
    wandbarg : dict : Arguments pour wandb.init() si wandb est utilisé, sinon None.
    save_model : bool : Si True, enregistre le modèle après l'entraînement.
    """
    run = None
    if wandbarg is not None:
        import wandb
        #wandb.login(key="c656dc47be1ed8b7866027b0569dca27b78821d9")  # Remplacez par votre clé API
        run = wandb.init(mode = "offline", **wandbarg)
    
    # Chargement du train dataset
    if verbose:
        print(f"Loading train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")
    try:
        train_set = ImportDataset(f"datasets1000/train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")
    except FileNotFoundError:
        print(f"File not found. Generating dataset with {num_data_train} data.")
        return
    
    if verbose:
        print(f"Loaded test_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")
    try:
        test_set = ImportDataset(f"datasets1000/test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt")
    except FileNotFoundError:
        print(f"File not found. Generating dataset with {num_data_train} data.")
        return
   
    # Construction du dataloader
    train_loader = train_set.get_dataloader(batch_size=batch_size, shuffle=True)
    test_loader = test_set.get_dataloader(batch_size=batch_size, shuffle=False)

    # Paramètres du problème
    weights = train_set.get_weights(tensor=True)
    capacities = train_set.get_capacities(tensor=True)

    # Modèle, optimiseur et scheduleur
    optimizer = optim.Adam(model.parameters(), lr)
    if schedulerType == "None":
        scheduler = None
    elif schedulerType == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, sched_step_size, sched_gamma)
    elif schedulerType == "ReduceLROnPlateau":
        if jobtype == "LD" or jobtype == "SG": patience = 15
        else: patience = 3
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True)
    elif schedulerType == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, div_factor= 10,final_div_factor=1e1,steps_per_epoch=len(train_loader), epochs=epochs)
    

    # Entraînement
    if jobtype == "LD":
        if verbose:
            print("Training the model with LD bound as loss...")
        train_LD(model, run, train_loader, test_loader, optimizer, scheduler, weights, capacities, epochs,
                 IMLE_n_samples=IMLE_n_samples, IMLE_sigma=IMLE_sigma, IMLE_lambd=IMLE_lambd, IMLE_two_sides=False, IMLE_processes=IMLE_processes,
                 verbose=verbose)
    elif jobtype == "classic":
        if verbose:
            print("Training the model with regret as loss...")
        train(model, run, train_loader, test_loader, optimizer, scheduler, weights, capacities, epochs, 
                    IMLE_n_samples=IMLE_n_samples, IMLE_sigma=IMLE_sigma, IMLE_lambd=IMLE_lambd, IMLE_two_sides=False, IMLE_processes=IMLE_processes,
                    verbose=verbose)
    elif jobtype == "SG":
        if verbose:
            print("Training the model with dynamic mu and LD bound as loss...")
        train_SG(model, run, train_loader, test_loader, optimizer, scheduler, weights, capacities, epochs,
                 IMLE_n_samples=IMLE_n_samples, IMLE_sigma=IMLE_sigma, IMLE_lambd=IMLE_lambd, IMLE_two_sides=False, IMLE_processes=IMLE_processes,
                 verbose=verbose, step_mu=step_mu, n_iter_mu=n_iter_mu)
        


    # Enregistrement du modèle
    if save_model:
        if jobtype == "LD":
            if verbose:
                print("Saving the model to models/LD_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'models/LD_{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
        elif jobtype == "classic":
            if verbose:
                print("Saving the model to models/{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'models/{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
        elif jobtype == "SG":
            if verbose:
                print("Saving the model to models/SG_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'models/SG_{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
    
    # Fin de l'exécution
    if run is not None:
        run.finish()


### EXÉCUTION DES EXPÉRIENCES ###
args = parser.parse_args()

#Choix des dimensions du problème
num_feat = 200
num_data_train = 500 # Taille du dataset d'entraînement
num_data_test = 100 # Taille du dataset de test

epochs_LD = args.ep_ld
epochs_classic = args.ep_cla
epochs_SG = args.ep_sg
lr_LD = 0.001
lr_classic = 0.001
lr_SG = 0.002
IMLE_n_samples_LD = 10
IMLE_n_samples_classic = 10
IMLE_sigma_LD = 1
IMLE_sigma_classic = 1
IMLE_lambd_LD = 10
IMLE_lambd_classic = 10
schedulerType_LD = "OneCycleLR" # "StepLR", "ReduceLROnPlateau", "OneCycleLR", "None"
schedulerType_classic = "StepLR" # "StepLR", "ReduceLROnPlateau", "OneCycleLR", "None"
schedulerType_SG = "StepLR" # "StepLR", "ReduceLROnPlateau", "OneCycleLR", "None"
IMLE_processes_LD = 1
IMLE_processes_classic = 1
dropout = 0.2


d = args.dim
n = args.n

# Choix dimension modèle
hidden_layer = 100

print(f"Entrainement sur {epochs_classic} epochs pour le modèle classique, {epochs_LD} epochs pour le modèle LD et {epochs_SG} pour le modèle avec SG de mu sur {d} contraintes et {n} items.")


dim = [args.dim]
num_item = [args.n]
for d in dim:
    for n in num_item:
        ### AVEC LD ###
        model = CustomMLP([num_feat,hidden_layer, n], dropout=dropout).to(device)
        wandbarg = {
                'entity': "hugoper-polytechnique-montr-al",
                'project': "DFL_LD",
                'dir': "./",
                'name': f"LD_{d}_{num_feat}_{n}_{num_data_train}",
                'group': f"{d}_{num_feat}_{n}_{num_data_train}",
                'job_type': "LD",
                'config': {
                    "architecture": f"MLP_{[num_feat, hidden_layer, n]}",
                    "dropout": dropout,
                    "dataset_train": f"train_{d}_{num_feat}_{n}_{num_data_train}.txt",
                    "dataset_test": f"test_{d}_{num_feat}_{n}_{num_data_test}.txt",
                    "batch_size": 32,
                    "epochs": epochs_LD,
                    "learning_rate": lr_LD,
                    "schedulerType": schedulerType_LD,
                    "sched_step_size": 50,
                    "sched_gamma": 0.5,
                    "IMLE_n_samples": IMLE_n_samples_LD,
                    "IMLE_sigma": IMLE_sigma_LD,
                    "IMLE_lambd": IMLE_lambd_LD,
                    "IMLE_two_sides": False,
                    "IMLE_processes": IMLE_processes_LD,
                }
        }
        if epochs_LD > 0:
            run_train(model, "LD", d, num_feat, n, num_data_train, num_data_test, epochs=epochs_LD, lr=lr_LD,schedulerType=schedulerType_LD, verbose=True, wandbarg=wandbarg,
                    IMLE_n_samples=IMLE_n_samples_LD, IMLE_sigma=IMLE_sigma_LD, IMLE_lambd=IMLE_lambd_LD, IMLE_processes=IMLE_processes_LD)
        
        ### SANS LD ###
        model = CustomMLP([num_feat, hidden_layer, n], dropout=dropout).to(device)
        wandbarg = {
                'entity': "hugoper-polytechnique-montr-al",
                'project': "DFL_LD",
                'dir': "./",
                'name': f"classic_{d}_{num_feat}_{n}_{num_data_train}",
                'group': f"{d}_{num_feat}_{n}_{num_data_train}",
                'job_type': "classic",
                'config': {
                    "architecture": f"MLP_{[num_feat, hidden_layer, n]}",
                    "dropout": dropout,
                    "dataset_train": f"train_{d}_{num_feat}_{n}_{num_data_train}.txt",
                    "dataset_test": f"test_{d}_{num_feat}_{n}_{num_data_test}.txt",
                    "batch_size": 32,
                    "epochs": epochs_classic,
                    "learning_rate": lr_classic,
                    "schedulerType": schedulerType_classic,
                    "sched_step_size": 10,
                    "sched_gamma": 0.1,
                    "IMLE_n_samples": IMLE_n_samples_classic,
                    "IMLE_sigma": IMLE_sigma_classic,
                    "IMLE_lambd": IMLE_lambd_classic,
                    "IMLE_two_sides": False,
                    "IMLE_processes": IMLE_processes_classic,
                }
        }
        if epochs_classic > 0:
            run_train(model, "classic", d, num_feat, n, num_data_train, num_data_test, epochs=epochs_classic, lr=lr_classic,schedulerType=schedulerType_classic, verbose=True, wandbarg=wandbarg,
                    IMLE_n_samples=IMLE_n_samples_classic, IMLE_sigma=IMLE_sigma_classic, IMLE_lambd=IMLE_lambd_classic, IMLE_processes=IMLE_processes_classic)
            
        ### MU DYNAMIQUE ###
        model = CustomMLP([num_feat, hidden_layer, n], dropout=dropout).to(device)
        wandbarg = {
                'entity': "hugoper-polytechnique-montr-al",
                'project': "DFL_LD",
                'dir': "./",
                'name': f"SG_{d}_{num_feat}_{n}_{num_data_train}",
                'group': f"{d}_{num_feat}_{n}_{num_data_train}",
                'job_type': "SG",
                'config': {
                    "architecture": f"MLP_{[num_feat, hidden_layer, n]}",
                    "dropout": dropout,
                    "dataset_train": f"train_{d}_{num_feat}_{n}_{num_data_train}.txt",
                    "dataset_test": f"test_{d}_{num_feat}_{n}_{num_data_test}.txt",
                    "batch_size": 32,
                    "epochs": epochs_SG,
                    "learning_rate": lr_SG,
                    "schedulerType": schedulerType_SG,
                    "sched_step_size": 150,
                    "sched_gamma": 0.5,
                    "IMLE_n_samples": IMLE_n_samples_classic,
                    "IMLE_sigma": IMLE_sigma_classic,
                    "IMLE_lambd": IMLE_lambd_classic,
                    "IMLE_two_sides": False,
                    "IMLE_processes": IMLE_processes_classic,
                    "step_mu": args.step_mu,
                    "n_iter_mu": args.n_iter_mu
                }
        }

        if epochs_SG > 0:
            run_train(model, "SG", d, num_feat, n, num_data_train, num_data_test, epochs=epochs_SG, lr=lr_SG,schedulerType=schedulerType_SG, verbose=True, wandbarg=wandbarg,
                    IMLE_n_samples=IMLE_n_samples_classic, IMLE_sigma=IMLE_sigma_classic, IMLE_lambd=IMLE_lambd_classic, IMLE_processes=IMLE_processes_classic, step_mu=args.step_mu, n_iter_mu=args.n_iter_mu,
                    sched_step_size=500, sched_gamma=0.5)
