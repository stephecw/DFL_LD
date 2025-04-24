import torch
from torch import optim
from data_import import ImportDataset
from train_imle import train, train_LD
import train_imle
from train_imle import LinearRegression, CustomMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Entraînement sur :", device)

def run_train(model, LD, dim, num_feat, num_item, num_data_train, num_data_test, 
              batch_size=32, epochs=20, lr=1e-3, 
              schedulerType="StepLR", sched_step_size=10, sched_gamma=0.1,
              IMLE_n_samples=10, IMLE_sigma=1.0, IMLE_lambd=10, IMLE_two_sides=False, IMLE_processes=1,
              verbose=False, wandbarg=None, save_model=True):
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
        train_set = ImportDataset(f"datasets/train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")
    except FileNotFoundError:
        print(f"File not found. Generating dataset with {num_data_train} data.")
        return
    
    if verbose:
        print(f"Loaded test_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")
    try:
        test_set = ImportDataset(f"datasets/test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt")
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, sched_step_size, sched_gamma)

    # Entraînement
    if LD:
        if verbose:
            print("Training the model with LD bound as loss...")
        train_LD(model, run, train_loader, test_loader, optimizer, scheduler, weights, capacities, epochs,
                 IMLE_n_samples=10, IMLE_sigma=1.0, IMLE_lambd=10, IMLE_two_sides=False, IMLE_processes=1,
                 verbose=verbose)
    else:
        if verbose:
            print("Training the model with regret as loss...")
        train(model, run, train_loader, test_loader, optimizer, scheduler, weights, capacities, epochs, 
                    IMLE_n_samples=10, IMLE_sigma=1.0, IMLE_lambd=10, IMLE_two_sides=False, IMLE_processes=1,
                    verbose=verbose)
    
    # Enregistrement du modèle
    if save_model:
        if LD:
            if verbose:
                print("Saving the model to models/LD_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'models/LD_{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
        else:
            if verbose:
                print("Saving the model to models/{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'models/{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
    
    # Fin de l'exécution
    if run is not None:
        run.finish()


### EXÉCUTION DES EXPÉRIENCES ###

#Choix des dimensions du problème
num_feat = 200
num_data_train = 500 # Taille du dataset d'entraînement
num_data_test = 100 # Taille du dataset de test
lr = 0.001
epochs_LD = 200
epochs = 20

# Choix dimension modèle
hidden_layer = 100

dim = [10]# [5, 10]
num_item = [30] #[30, 50, 100]
for d in dim:
    for n in num_item:
        ### AVEC LD ###
        model = CustomMLP([num_feat, hidden_layer, n]).to(device)
        wandbarg = {
                'entity': "hugoper-polytechnique-montr-al",
                'project': "DFL_LD",
                'dir': "./",
                'name': f"LD_{d}_{num_feat}_{n}_{num_data_train}",
                'group': f"{d}_{num_feat}_{n}_{num_data_train}",
                'job_type': "LD",
                'config': {
                    "architecture": f"MLP_{[num_feat, hidden_layer, num_item]}",
                    "dataset_train": f"train_{d}_{num_feat}_{n}_{num_data_train}.txt",
                    "dataset_test": f"test_{d}_{num_feat}_{n}_{num_data_test}.txt",
                    "batch_size": 32,
                    "epochs": epochs_LD,
                    "learning_rate": lr,
                    "schedulerType": "None",
                    "sched_step_size": 10,
                    "sched_gamma": 0.1,
                    "IMLE_n_samples": 10,
                    "IMLE_sigma": 1.0,
                    "IMLE_lambd": 10,
                    "IMLE_two_sides": False,
                    "IMLE_processes": 1,
                }
        }
        run_train(model, True, d, num_feat, n, num_data_train, num_data_test, epochs=epochs_LD, lr=lr,schedulerType=None, verbose=True, wandbarg=wandbarg)
        
        ### SANS LD ###
        model = CustomMLP([num_feat, hidden_layer, n]).to(device)
        wandbarg = {
                'entity': "hugoper-polytechnique-montr-al",
                'project': "DFL_LD",
                'dir': "./",
                'name': f"classic_{d}_{num_feat}_{n}_{num_data_train}",
                'group': f"{d}_{num_feat}_{n}_{num_data_train}",
                'job_type': "classic",
                'config': {
                    "architecture": f"MLP_{[num_feat, hidden_layer, num_item]}",
                    "dataset_train": f"train_{d}_{num_feat}_{n}_{num_data_train}.txt",
                    "dataset_test": f"test_{d}_{num_feat}_{n}_{num_data_test}.txt",
                    "batch_size": 32,
                    "epochs": epochs,
                    "learning_rate": lr,
                    "schedulerType": "None",
                    "sched_step_size": 10,
                    "sched_gamma": 0.1,
                    "IMLE_n_samples": 10,
                    "IMLE_sigma": 1.0,
                    "IMLE_lambd": 10,
                    "IMLE_two_sides": False,
                    "IMLE_processes": 1,
                }
        }
        run_train(model, False, d, num_feat, n, num_data_train, num_data_test, epochs=epochs, lr=lr,schedulerType=None, verbose=True, wandbarg=wandbarg)