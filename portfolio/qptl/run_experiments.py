import sched
import torch
from torch import optim
from data_import import ImportDataset
from qptl.train_qptl import train, train_LD, train_SG

from qptl.train_qptl import CustomMLP

import argparse

# Définir les arguments de ligne de commande
parser = argparse.ArgumentParser(description="Script d'entraînement avec des dimensions spécifiées.")
parser.add_argument('--n', type=int, default=30, help='Nombre d\'item.')
parser.add_argument('--ep_cla', type=int, default=0, help='Nombre d\'epochs pour l\'entraînement classique. (0 pour ne pas l\'exécuter)')
parser.add_argument('--ep_ld', type=int, default=0, help='Nombre d\'epochs pour l\'entraînement LD. (0 pour ne pas l\'exécuter)')
parser.add_argument('--ep_sg', type=int, default=0, help='Nombre d\'epochs pour l\'entraînement SG. (0 pour ne pas l\'exécuter)')
parser.add_argument('--step_mu', type=int, default=5, help='Nombre d\'epochs entre la mise à jour des \mu. (0 pour ne pas l\'exécuter)')
parser.add_argument('--n_iter_mu', type=int, default=10, help='Nombre d\'itérations pour l\'optimisation de \mu. (0 pour ne pas l\'exécuter)')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Entraînement sur :", device)

def run_train(model, jobtype, gamma, num_feat, num_item, num_data_train, num_data_test, 
              batch_size=32, epochs=20, lr=1e-3, 
              schedulerType="StepLR", sched_step_size=50, sched_gamma=0.5,
              QPTL_alpha =1e-6, QPTL_regularizer='quadratic',
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
    if verbose:
        print(f"Loading train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt")
    try:
        train_set = ImportDataset(f"datasets/train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt")
    except FileNotFoundError:
        print(f"File not found. Generating dataset with {num_data_train} data.")
        return
    
    if verbose:
        print(f"Loading test_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt")
    try:
        test_set = ImportDataset(f"datasets/test_{num_item}_{num_data_test}_{num_feat}_{gamma_str}.txt")
    except FileNotFoundError:
        print(f"File not found. Generating dataset with {num_data_test} data.")
        return
   
    # Construction du dataloader
    train_loader = train_set.get_dataloader(batch_size=batch_size, shuffle=True)
    test_loader = test_set.get_dataloader(batch_size=batch_size, shuffle=False)

    # Paramètres du problème
    cov = train_set.get_cov()


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
        train_LD(model, run, train_loader, test_loader, optimizer, scheduler, cov, gamma, epochs,
                 QPTL_alpha=QPTL_alpha, regularized=QPTL_regularizer,
                 verbose=verbose)
    elif jobtype == "classic":
        if verbose:
            print("Training the model with regret as loss...")
        train(model, run, train_loader, test_loader, optimizer, scheduler, cov, gamma, epochs, 
                    QPTL_alpha=QPTL_alpha, regularized=QPTL_regularizer,
                    verbose=verbose)
    elif jobtype == "SG":
        if verbose:
            print("Training the model with dynamic mu and LD bound as loss...")
        train_SG(model, run, train_loader, test_loader, optimizer, scheduler, cov, gamma, epochs,
                QPTL_alpha=QPTL_alpha, regularized=QPTL_regularizer,
                 verbose=verbose, step_mu=step_mu, n_iter_mu=n_iter_mu)
        


    # Enregistrement du modèle
    if save_model:
        if jobtype == "LD":
            if verbose:
                print("Saving the model to models/LD_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth")
            torch.save(model.state_dict(), f'qptl/models/LD_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth')
        elif jobtype == "classic":
            if verbose:
                print("Saving the model to models/{num_item}_{num_data_train}_{num_feat}.pth")
            torch.save(model.state_dict(), f'qptl/models/{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth')
        elif jobtype == "SG":
            if verbose:
                print("Saving the model to models/SG_{num_item}_{num_data_train}_{num_feat}.pth")
            torch.save(model.state_dict(), f'qptl/models/SG_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth')
    
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

epochs_LD = args.ep_ld
epochs_classic = args.ep_cla
epochs_SG = args.ep_sg
lr_LD = 0.001
lr_classic = 0.001
lr_SG = 0.001
QPTL_alpha = 1e-6
QPTL_regularizer = 'quadratic'
schedulerType_LD = "OneCycleLR" # "StepLR", "ReduceLROnPlateau", "OneCycleLR", "None"
schedulerType_classic = "StepLR" # "StepLR", "ReduceLROnPlateau", "OneCycleLR", "None"
schedulerType_SG = "StepLR" # "StepLR", "ReduceLROnPlateau", "OneCycleLR", "None"
dropout = 0.2


n = args.n

# Choix dimension modèle
hidden_layer = 100

print(f"Entrainement sur {epochs_classic} epochs pour le modèle classique, {epochs_LD} epochs pour le modèle LD et {epochs_SG} pour le modèle avec SG de mu sur {n} items.")


num_item = [args.n]
for n in num_item:
    ### AVEC LD ###
    model = CustomMLP([num_feat,hidden_layer, n], dropout=dropout).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"LD_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'group': f"portfolio_qptl_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'job_type': "LD",
            'config': {
                "architecture": f"MLP_{[num_feat, hidden_layer, n]}",
                "dropout": dropout,
                "dataset_train": f"train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",                    "dataset_test": f"test_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "batch_size": 32,
                "epochs": epochs_LD,
                "learning_rate": lr_LD,
                "schedulerType": schedulerType_LD,
                "sched_step_size": 50,
                "sched_gamma": 0.5,
                "QPTL_alpha": QPTL_alpha,
                "QPTL_regularizer": QPTL_regularizer,
            }
    }
    if epochs_LD > 0:
        run_train(model, "LD", gamma, num_feat, n, num_data_train, num_data_test, epochs=epochs_LD, lr=lr_LD,schedulerType=schedulerType_LD, verbose=True, wandbarg=wandbarg,
                QPTL_alpha=QPTL_alpha, QPTL_regularizer=QPTL_regularizer)
        
    ### SANS LD ###
    model = CustomMLP([num_feat, hidden_layer, n], dropout=dropout).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"classic_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'group': f"portfolio_qptl_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'job_type': "classic",
            'config': {
                "architecture": f"MLP_{[num_feat, hidden_layer, n]}",
                "dropout": dropout,
                "dataset_train": f"train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "dataset_test": f"test_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "batch_size": 32,
                "epochs": epochs_classic,
                "learning_rate": lr_classic,
                "schedulerType": schedulerType_classic,
                "sched_step_size": 10,
                "sched_gamma": 0.1,
                "QPTL_alpha": QPTL_alpha,
                "QPTL_regularizer": QPTL_regularizer,
            }
    }
    if epochs_classic > 0:
        run_train(model, "classic", gamma, num_feat, n, num_data_train, num_data_test, epochs=epochs_classic, lr=lr_classic,schedulerType=schedulerType_classic, verbose=True, wandbarg=wandbarg,
                QPTL_alpha=QPTL_alpha, QPTL_regularizer=QPTL_regularizer, sched_step_size=100, sched_gamma=0.5)
            
    ### MU DYNAMIQUE ###
    model = CustomMLP([num_feat, hidden_layer, n], dropout=dropout).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"SG_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'group': f"portfolio_qptl_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'job_type': "SG",
            'config': {
                "architecture": f"MLP_{[num_feat, hidden_layer, n]}",
                "dropout": dropout,
                "dataset_train": f"train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "dataset_test": f"test_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "batch_size": 32,
                "epochs": epochs_SG,
                "learning_rate": lr_SG,
                "schedulerType": schedulerType_SG,
                "sched_step_size": 150,
                "sched_gamma": 0.5,
                "QPTL_alpha": QPTL_alpha,
                "QPTL_regularizer": QPTL_regularizer,
                "step_mu": args.step_mu,
                "n_iter_mu": args.n_iter_mu
            }
    }

    if epochs_SG > 0:
        run_train(model, "SG", gamma, num_feat, n, num_data_train, num_data_test, epochs=epochs_SG, lr=lr_SG,schedulerType=schedulerType_SG, verbose=True, wandbarg=wandbarg,
                QPTL_alpha=QPTL_alpha, QPTL_regularizer=QPTL_regularizer, step_mu=args.step_mu, n_iter_mu=args.n_iter_mu,
                sched_step_size=100, sched_gamma=0.5)
