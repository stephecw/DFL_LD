import sched
import torch
from torch import optim
from data_import import ImportDataset
from train import train, train_LD, train_SG

from train import CustomMLP

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
              batch_size, epochs, lr, 
              schedulerType, sched_arg,
              diff_method, diff_method_arg,
              step_mu=5, num_iter_mu=15,
              verbose=False, wandbarg=None, time_limit=None, save_model=True):
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
    scheduler = None
    if schedulerType == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, **sched_arg)
    elif schedulerType == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_arg)
    elif schedulerType == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **sched_arg)
    

    # Entraînement
    if jobtype == "LD":
        if verbose:
            print("Training the model with LD bound as loss...")
        train_LD(model, run, train_loader, test_loader, optimizer, scheduler, weights, capacities, epochs, time_limit,
                 diff_method, diff_method_arg,
                 verbose=verbose)
    elif jobtype == "classic":
        if verbose:
            print("Training the model with regret as loss...")
        train(  model, run, train_loader, test_loader, optimizer, scheduler, weights, capacities, epochs, time_limit,
                diff_method, diff_method_arg,
                verbose=verbose)
    elif jobtype == "SG":
        if verbose:
            print("Training the model with dynamic mu and LD bound as loss...")
        train_SG(model, run, train_loader, test_loader, optimizer, scheduler, weights, capacities, epochs, time_limit,
                 diff_method, diff_method_arg,
                 step_mu=step_mu, num_iter_mu=num_iter_mu,
                 verbose=verbose)
        


    # Enregistrement du modèle
    if save_model:
        if jobtype == "LD":
            if verbose:
                print(f"Saving the model to models/{diff_method}_LD_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'models/{diff_method}_LD_{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
        elif jobtype == "classic":
            if verbose:
                print(f"Saving the model to models/{diff_method}_classic_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'models/{diff_method}_classic_{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
        elif jobtype == "SG":
            if verbose:
                print(f"Saving the model to models/{diff_method}_SG_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'models/{diff_method}_SG_{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
    
    # Fin de l'exécution
    if run is not None:
        run.finish()


### EXÉCUTION DES EXPÉRIENCES ###
args = parser.parse_args()

#Choix des dimensions du problème
num_feat = 200
num_data_train = 500 # Taille du dataset d'entraînement
num_data_test = 100 # Taille du dataset de test
dim = args.dim
num_item = args.n

# Paramètre classique 
epochs_classic = args.ep_cla
time_limit_classic = 1800
batch_size_classic = 32
lr_classic = 0.001
model_shape_classic = [num_feat, 100, num_item]
dropout_classic = 0.2
schedulerType_classic = None # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg_classic = {'step_size':100,
                     'gamma':0.5
                     }
diff_method_classic = "SPOPlus" # "StepLR", "SPOPlus"
diff_method_arg_classic = { }

# Paramètre LD
epochs_LD = args.ep_ld
time_limit_LD = 1800
batch_size_LD = 32
lr_LD = 0.001
model_shape_LD = [num_feat, 100, num_item]
dropout_LD = 0.2
schedulerType_LD = None # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg_LD = {'step_size':100,
                     'gamma':0.5
                     }
diff_method_LD= "SPOPlus" # "IMLE", "SPOPlus"
diff_method_arg_LD = { }

# Paramètre SG
epochs_SG = args.ep_sg
time_limit_SG = 3600
batch_size_SG = 32
lr_SG = 0.001
model_shape_SG = [num_feat, 100, num_item]
dropout_SG = 0.2
schedulerType_SG = None # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg_SG = {'step_size':100,
                     'gamma':0.5
                     }
diff_method_SG= "SPOPlus" # "IMLE", "SPOPlus"
diff_method_arg_SG = {}
step_mu = args.step_mu
num_iter_mu = args.n_iter_mu

print(f"Entrainement sur {epochs_classic} epochs pour le modèle classique, {epochs_LD} epochs pour le modèle LD et {epochs_SG} pour le modèle avec SG de mu sur {dim} contraintes et {num_item} items.")

### EXECUTION ###
## LD ##
if epochs_LD > 0:
    model = CustomMLP(model_shape_LD, dropout=dropout_LD).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"{diff_method_LD}_LD_{dim}_{num_feat}_{num_item}_{num_data_train}",
            'group': f"présentation_05_11",
            'job_type': "LD",
            'config': {
                "architecture": model_shape_LD,
                "dropout": dropout_LD,
                "dataset_train": f"train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt",
                "dataset_test": f"test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt",
                "batch_size": batch_size_LD,
                "epochs": epochs_LD,
                "time_limit":time_limit_LD,
                "learning_rate": lr_LD,
                "schedulerType": schedulerType_LD,
                "sched_arg":sched_arg_LD,
                "diff_method":diff_method_LD,
                "diff_method_arg":diff_method_arg_LD
            }
    }
    run_train(model, "LD", dim, num_feat, num_item, num_data_train, num_data_test,
            batch_size=batch_size_LD, epochs=epochs_LD, lr=lr_LD, 
            schedulerType=schedulerType_LD, sched_arg=sched_arg_LD,
            diff_method=diff_method_LD, diff_method_arg=diff_method_arg_LD,
            verbose=True, wandbarg=wandbarg, time_limit=time_limit_LD)

## Classic ##
if epochs_classic > 0:
    model = CustomMLP(model_shape_classic, dropout=dropout_classic).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"{diff_method_classic}_Classic_{dim}_{num_feat}_{num_item}_{num_data_train}",
            'group': f"présentation_05_11",
            'job_type': "Classic",
            'config': {
                "architecture": model_shape_classic,
                "dropout": dropout_classic,
                "dataset_train": f"train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt",
                "dataset_test": f"test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt",
                "batch_size": batch_size_classic,
                "epochs": epochs_classic,
                "time_limit":time_limit_classic,
                "learning_rate": lr_classic,
                "schedulerType": schedulerType_classic,
                "sched_arg":sched_arg_classic,
                "diff_method":diff_method_classic,
                "diff_method_arg":diff_method_arg_classic
            }
    }
    run_train(model, "classic", dim, num_feat, num_item, num_data_train, num_data_test,
            batch_size=batch_size_classic, epochs=epochs_classic, lr=lr_classic, 
            schedulerType=schedulerType_classic, sched_arg=sched_arg_classic,
            diff_method=diff_method_classic, diff_method_arg=diff_method_arg_classic,
            verbose=True, wandbarg=wandbarg, time_limit=time_limit_classic)
    
if epochs_SG > 0:
    model = CustomMLP(model_shape_SG, dropout=dropout_SG).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"{diff_method_SG}_SG_{dim}_{num_feat}_{num_item}_{num_data_train}",
            'group': f"présentation_05_11",
            'job_type': "SG",
            'config': {
                "architecture": model_shape_SG,
                "dropout": dropout_SG,
                "dataset_train": f"train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt",
                "dataset_test": f"test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt",
                "batch_size": batch_size_SG,
                "epochs": epochs_SG,
                "time_limit":time_limit_SG,
                "learning_rate": lr_SG,
                "schedulerType": schedulerType_SG,
                "sched_arg":sched_arg_SG,
                "diff_method":diff_method_SG,
                "diff_method_arg":diff_method_arg_SG,
                "step_mu":step_mu,
                "num_iter_mu":num_iter_mu
            }
    }
    run_train(model, "SG", dim, num_feat, num_item, num_data_train, num_data_test,
            batch_size=batch_size_SG, epochs=epochs_SG, lr=lr_SG, 
            schedulerType=schedulerType_SG, sched_arg=sched_arg_SG,
            diff_method=diff_method_SG, diff_method_arg=diff_method_arg_SG,
            step_mu=step_mu, num_iter_mu=num_iter_mu,
            verbose=True, wandbarg=wandbarg, time_limit=time_limit_SG)