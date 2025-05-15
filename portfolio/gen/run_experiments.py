import torch
from torch import optim
from data_import import ImportDataset
from gen.train_gen import train, train_LD, train_SG
import numpy as np

from gen.train_gen import CustomMLP

import argparse

# Définir les arguments de ligne de commande
parser = argparse.ArgumentParser(description="Script d'entraînement avec des dimensions spécifiées.")
parser.add_argument('--n', type=int, default=30, help='Nombre d\'item.')
parser.add_argument('--ep_cla', type=int, default=0, help='Nombre d\'epochs pour l\'entraînement classique. (0 pour ne pas l\'exécuter)')
parser.add_argument('--ep_ld', type=int, default=0, help='Nombre d\'epochs pour l\'entraînement LD. (0 pour ne pas l\'exécuter)')
parser.add_argument('--ep_sg', type=int, default=0, help='Nombre d\'epochs pour l\'entraînement SG. (0 pour ne pas l\'exécuter)')
parser.add_argument('--step_mu', type=int, default=5, help='Nombre d\'epochs entre la mise à jour des \mu. (0 pour ne pas l\'exécuter)')
parser.add_argument('--n_iter_mu', type=int, default=10, help='Nombre d\'itérations pour l\'optimisation de \mu. (0 pour ne pas l\'exécuter)')
parser.add_argument(
    '--method',
    type=str,
    default='imle',
    help="Méthode d'optimisation (ex: imle, reinforce, sample)"
    )
parser.add_argument('--lin', type=int, default=1, help='1 pour prendre la contrainte linéraire pour le sous-prob principal, 0 pour la contrainte quadratique')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Entraînement sur :", device)


def run_train(model, method, jobtype, gamma, num_feat, num_item, num_data_train, num_data_test, principal_lin=True,
              batch_size=32, epochs=20, lr=1e-3, 
              schedulerType="StepLR", sched_step_size=50, sched_gamma=0.5,
              IMLE_n_samples=10, IMLE_sigma=1.0, IMLE_lambd=10, IMLE_two_sides=False, IMLE_processes=1,
              SPO_solve_ratio = 1, SPO_reduction = 'mean',SPO_processes = 1,
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
    gamma_str = str(gamma).replace('.', '-')
    fold = "/lin" if principal_lin else "/quad"
    if verbose:
        print(f"Loading {fold}/train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt")
    try:
        train_set = ImportDataset(f"datasets{fold}/train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt")
    except FileNotFoundError:
        print(f"File not found.")
        return
    
    if verbose:
        print(f"Loading {fold}/test_{num_item}_{num_data_test}_{num_feat}_{gamma_str}.txt")
    try:
        test_set = ImportDataset(f"datasets{fold}/test_{num_item}_{num_data_test}_{num_feat}_{gamma_str}.txt")
    except FileNotFoundError:
        print(f"File not found.")
        return
   
    # Construction du dataloader
    train_loader = train_set.get_dataloader(batch_size=batch_size, shuffle=True)
    test_loader = test_set.get_dataloader(batch_size=batch_size, shuffle=False)

    # Paramètres du problème
    cov = train_set.get_cov()

    def is_positive_definite_cholesky(cov: np.ndarray) -> bool:
        """
        envoie True si cov est définie positive (PD) au sens strict,
        False sinon (semi‑déf. ou indéfinie).
        """
        try:
            np.linalg.cholesky(cov)
            return True
        except np.linalg.LinAlgError:
            return False
    
    if not is_positive_definite_cholesky(cov):
        print("La matrice de covariance n'est pas définie positive.")

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
        train_LD(model, method, run, train_loader, test_loader, optimizer, scheduler, cov, gamma, epochs, principal_lin,
                    IMLE_n_samples=IMLE_n_samples, IMLE_sigma=IMLE_sigma, IMLE_lambd=IMLE_lambd, IMLE_two_sides=False, IMLE_processes=IMLE_processes,
                    SPO_solve_ratio=SPO_solve_ratio, SPO_reduction=SPO_reduction, SPO_processes=SPO_processes,
                    verbose=verbose)
    elif jobtype == "classic":
        if verbose:
            print("Training the model with regret as loss...")
        train(model, method, run, train_loader, test_loader, optimizer, scheduler, cov, gamma, epochs,
                    IMLE_n_samples=IMLE_n_samples, IMLE_sigma=IMLE_sigma, IMLE_lambd=IMLE_lambd, IMLE_two_sides=False, IMLE_processes=IMLE_processes,
                    SPO_solve_ratio=SPO_solve_ratio, SPO_reduction=SPO_reduction, SPO_processes=SPO_processes,
                    verbose=verbose)
    elif jobtype == "SG":
        if verbose:
            print("Training the model with dynamic mu and LD bound as loss...")

        train_SG(model, method, run, train_loader, test_loader, optimizer, scheduler, cov, gamma, epochs, principal_lin,
                    IMLE_n_samples=IMLE_n_samples, IMLE_sigma=IMLE_sigma, IMLE_lambd=IMLE_lambd, IMLE_two_sides=False, IMLE_processes=IMLE_processes,
                    SPO_solve_ratio=SPO_solve_ratio, SPO_reduction=SPO_reduction, SPO_processes=SPO_processes,
                    step_mu=step_mu, n_iter_mu=n_iter_mu,
                    verbose=verbose)

        


    # Enregistrement du modèle
    if save_model:
        if jobtype == "LD":
            if verbose:
                print("Saving the model to models/LD_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'models/{method}_SG_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth')
        elif jobtype == "classic":
            if verbose:
                print("Saving the model to models/{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'models/{method}_SG_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth')
        elif jobtype == "SG":
            if verbose:
                print("Saving the model to models/SG_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'models/{method}_SG_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth')
    
    # Fin de l'exécution
    if run is not None:
        run.finish()


### EXÉCUTION DES EXPÉRIENCES ###
args = parser.parse_args()

#Choix des dimensions du problème
num_feat = 200
num_data_train = 500 # Taille du dataset d'entraînement
num_data_test = 100 # Taille du dataset de test
gamma = 2.25
gamma_str = str(gamma).replace('.', '-')
method = args.method
principal_lin = False if args.lin == 0 else True

epochs_LD = args.ep_ld
epochs_classic = args.ep_cla
epochs_SG = args.ep_sg
lr_LD = 0.001
lr_classic = 0.002
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


n = args.n

# Choix dimension modèle
hidden_layer = 100

print(f"Entrainement sur {epochs_classic} epochs pour le modèle classique, {epochs_LD} epochs pour le modèle LD et {epochs_SG} pour le modèle avec SG de mu sur {n} items.")
print(f"On utilise la méthode {method}")


num_item = [args.n]
for n in num_item:
    ### AVEC LD ###
    model = CustomMLP([num_feat,hidden_layer, n], dropout=dropout).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"{method}_LD_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'group': f"portfolio_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'job_type': "LD",
            'config': {
                "architecture": f"MLP_{[num_feat, hidden_layer, n]}",
                "dropout": dropout,
                "dataset_train": f"train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",                    
                "dataset_test": f"test_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "batch_size": 32,
                "gamma": gamma,
                "sp_principal": "lin" if principal_lin else "quad",
                "epochs": epochs_LD,
                "learning_rate": lr_LD,
                "schedulerType": schedulerType_LD,
                "sched_step_size": 100,
                "sched_gamma": 0.5,
            }
    }
    if epochs_LD > 0:
        run_train(model, method, "LD", gamma, num_feat, n, num_data_train, num_data_test, principal_lin=principal_lin, epochs=epochs_LD, lr=lr_LD,schedulerType=schedulerType_LD, verbose=True, wandbarg=wandbarg,
                  sched_step_size=100, sched_gamma=0.5)
        
    ### SANS LD ###
    model = CustomMLP([num_feat, hidden_layer, n], dropout=dropout).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"{method}_classic_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'group': f"portfolio_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'job_type': "classic",
            'config': {
                "architecture": f"MLP_{[num_feat, hidden_layer, n]}",
                "dropout": dropout,
                "dataset_train": f"train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "dataset_test": f"test_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "batch_size": 32,
                "gamma": gamma,
                "sp_principal": "lin" if principal_lin else "quad",
                "epochs": epochs_classic,
                "learning_rate": lr_classic,
                "schedulerType": schedulerType_classic,
                "sched_step_size": 40,
                "sched_gamma": 0.5
            }
    }
    if epochs_classic > 0:
        run_train(model, method, "classic", gamma, num_feat, n, num_data_train, num_data_test, principal_lin=principal_lin, epochs=epochs_classic, lr=lr_classic,schedulerType=schedulerType_classic, verbose=True, wandbarg=wandbarg,
        sched_step_size=40, sched_gamma=0.5)
            
    ### MU DYNAMIQUE ###
    model = CustomMLP([num_feat, hidden_layer, n], dropout=dropout).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"{method}_SG_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'group': f"portfolio_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'job_type': "SG",
            'config': {
                "architecture": f"MLP_{[num_feat, hidden_layer, n]}",
                "dropout": dropout,
                "dataset_train": f"train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "dataset_test": f"test_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "batch_size": 32,
                "gamma": gamma,
                "sp_principal": "lin" if principal_lin else "quad",
                "epochs": epochs_SG,
                "learning_rate": lr_SG,
                "schedulerType": schedulerType_SG,
                "sched_step_size": 300,
                "sched_gamma": 0.5,
                "step_mu": args.step_mu,
                "n_iter_mu": args.n_iter_mu
            }
    }

    if epochs_SG > 0:
        run_train(model, method, "SG", gamma, num_feat, n, num_data_train, num_data_test, principal_lin=principal_lin, epochs=epochs_SG, lr=lr_SG,schedulerType=schedulerType_SG, verbose=True, wandbarg=wandbarg,
                step_mu=args.step_mu, n_iter_mu=args.n_iter_mu,
                sched_step_size=300, sched_gamma=0.5)