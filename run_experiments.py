import torch
from torch import optim
from data_import import ImportDataset
from train_imle import train, test_regret, train_LD
import train_imle
from train_imle import LinearRegression, CustomMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Entraînement sur :", device)

def run_train(model, LD, dim, num_feat, num_item, num_data_train, epochs=20, lr=1e-3, verbose=False, wandbarg=None, save_model=True):
    """
    Fonction principale pour charger le dataset et entraîner le modèle.
    model : nn.Module : Modèle à entraîner.
    LD : bool : Si True, utilise la décomposition lagrangienne.
    dim : int : Nombre de dimensions.
    num_feat : int : Nombre de features.
    num_item : int : Nombre d'items.
    num_data_train : int : Nombre de données d'entraînement.
    epochs : int : Nombre d'époques d'entraînement.
    lr : float : Taux d'apprentissage.
    verbose : bool : Si True, affiche des informations sur le chargement du dataset.
    wandbarg : dict : Arguments pour wandb.init() si wandb est utilisé.
    save_model : bool : Si True, enregistre le modèle après l'entraînement.
    """
    run = None
    if wandbarg is not None:
        import wandb
        wandb.login(key="c656dc47be1ed8b7866027b0569dca27b78821d9")  # Remplacez par votre clé API
        run = wandb.init(mode = "offline", **wandbarg)
    
    # Chargement du train dataset
    if verbose:
        print(f"Loading train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")
    try:
        train_set = ImportDataset(f"datasets/train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")
    except FileNotFoundError:
        print(f"File not found. Generating dataset with {num_data_train} data.")
        return
   
    # Construction du dataloader
    train_loader = train_set.get_dataloader()

    # Paramètres du problème
    weights = train_set.get_weights(tensor=True).to(device)
    capacities = train_set.get_capacities(tensor=True).to(device)

    # Modèle, optimiseur et scheduleur
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Entraînement
    if LD:
        if verbose:
            print("Training the model with LD bound as loss...")
        train_LD(model, run, train_loader, optimizer, scheduler, weights, capacities, epochs, verbose)
    else:
        if verbose:
            print("Training the model with regret as loss...")
        train(model, run, train_loader, optimizer, scheduler, weights, capacities, epochs, verbose) 
    
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
    
def run_test(dim, num_feat, num_item, num_data_test, model, verbose=False, wandbarg=None):
    """
    Fonction principale pour charger le dataset de test et tester le modèle.
    dim : int : Nombre de dimensions.
    num_feat : int : Nombre de features.
    num_item : int : Nombre d'items.
    num_data_test : int : Nombre de données de test.
    model : nn.Module : Modèle entraîné.
    verbose : bool : Si True, affiche des informations sur le chargement du dataset.
    wandbarg : dict : Arguments pour wandb.init() si wandb est utilisé.
    """
    run = None
    if wandbarg is not None:
        import wandb
        wandb.login(key="c656dc47be1ed8b7866027b0569dca27b78821d9")  # Remplacez par votre clé API
        run = wandb.init(mode = "offline",**wandbarg)
    
    if verbose:
        print(f"Loading test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt")
    try:
        test_set = ImportDataset(f"datasets/test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt")
    except FileNotFoundError:
        print(f"File not found. Generating dataset with {num_data_test} data.")
        return 
       
    # Chargement du dataloader de test
    test_loader = test_set.get_dataloader()
    
    # Paramètres du problème
    weights = test_set.get_weights(tensor=True)
    capacities = test_set.get_capacities(tensor=True)
    
    if verbose:
        print("Testing the model...")
    test_regret(model, run, test_loader, weights, capacities, verbose)
    
    if run is not None:
        run.finish()
    


### EXÉCUTION DES EXPÉRIENCES ###

#Choix des dimensions du problème
num_feat = 200
num_data_train = 500 # Taille du dataset d'entraînement
num_data_test = 100 # Taille du dataset de test
lr = 0.001
epochs_LD = 200
epochs_classic = 20

# Choix dimension modèle
hidden_layer = 100

dim = [10]# [5, 10]
num_item = [30] #[30, 50, 100]
for d in dim:
    for n in num_item:
        ### AVEC LD ###
        model = CustomMLP([num_feat, hidden_layer, n]).to(device)
        wandbarg_train = {
                'entity': "hugoper-polytechnique-montr-al",
                'project': "DFL_LD",
                'dir': "./",
                'name': f"LD_train_{d}_{num_feat}_{n}_{num_data_train}",
                'group': f"{d}_{num_feat}_{n}_{num_data_train}",
                'job_type': "train_LD",
                'config': {
                    "learning_rate": lr,
                    "architecture": f"MLP_{[num_feat, hidden_layer, num_item]}",
                    "dataset": f"train_{d}_{num_feat}_{n}_{num_data_train}.txt",
                    "epochs": epochs_LD,
                }
        }
        run_train(model, True, d, num_feat, n, num_data_train, epochs_LD, lr, True, wandbarg_train)

        # Paramètres pour wandb, mettre à None si pas d'utilisation de wandb
        wandbarg_test = {
                'entity': "hugoper-polytechnique-montr-al",
                'project': "DFL_LD",
                'dir': "./",
                'name': f"LD_test_{d}_{num_feat}_{n}_{num_data_train}",
                'group': f"{d}_{num_feat}_{n}_{num_data_train}",
                'job_type': "test_LD",
                'config': {
                    "dataset": f"test_{d}_{num_feat}_{n}_{num_data_test}.txt",
                }
        }
        run_test(d, num_feat, n, num_data_test, model, True, wandbarg_test)
        
        ### SANS LD ###
        model = CustomMLP([num_feat, hidden_layer, n]).to(device)
        wandbarg_train = {
                'entity': "hugoper-polytechnique-montr-al",
                'project': "DFL_LD",
                'dir': "./",
                'name': f"classic_train_{d}_{num_feat}_{n}_{num_data_train}",
                'group': f"{d}_{num_feat}_{n}_{num_data_train}",
                'job_type': "train_classic",
                'config': {
                    "learning_rate": lr,
                    "architecture": f"MLP_{[num_feat, hidden_layer, num_item]}", # A changer si nécéssaire
                    "dataset": f"train_{d}_{num_feat}_{n}_{num_data_train}.txt",
                    "epochs": epochs_classic,
                }
        }
        run_train(model, False, d, num_feat, n, num_data_train, epochs_classic, lr, True, wandbarg_train)

        # Paramètres pour wandb, mettre à None si pas d'utilisation de wandb
        wandbarg_test = {
                'entity': "hugoper-polytechnique-montr-al",
                'project': "DFL_LD",
                'dir': "./",
                'name': f"classic_test_{d}_{num_feat}_{n}_{num_data_train}",
                'group': f"{d}_{num_feat}_{n}_{num_data_train}",
                'job_type': "test_classic",
                'config': {
                    "dataset": f"test_{d}_{num_feat}_{n}_{num_data_test}.txt",
                }
        }
        run_test(d, num_feat, n, num_data_test, model, True, wandbarg_test)