import torch
from torch import optim
import wandb
from data_import import ImportDataset
from train_imle import LinearRegression, train, test_regret, train_LD


def run_train(dim, num_feat, num_item, num_data_train, epochs=20, lr=1e-3, verbose=False, wandbarg=None):
    """
    Fonction principale pour charger le dataset, créer le modèle et l'entraîner.
    dim : int : Nombre de dimensions.
    num_feat : int : Nombre de features.
    num_item : int : Nombre d'items.
    num_data_train : int : Nombre de données d'entraînement.
    epochs : int : Nombre d'époques d'entraînement.
    lr : float : Taux d'apprentissage.
    verbose : bool : Si True, affiche des informations sur le chargement du dataset.
    wandbarg : dict : Arguments pour wandb.init() si wandb est utilisé.
    """
    run = None
    if wandb is not None:
        from wandb import wandb
        # Initialisation de wandb
        run = wandb.init(wandbarg)
    
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

    # Chargement du dataset
    fname_train = "datasets/train_5_20_20_5.txt"

    # Paramètres du problème
    weights = train_set.get_weights(tensor=True)
    capacities = train_set.get_capacities(tensor=True)

    # Modèle, optimiseur et scheduleur
    model = LinearRegression(num_feat, num_item) ## A CHANGER
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Entraînement
    if verbose:
        print("Training the model...")
    train(model, run, train_loader, optimizer, scheduler, weights, capacities, epochs, verbose)
    
    # Enregistrement du modèle
    torch.save(model.state_dict(), f'{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
    
    # Fin de l'exécution
    if run is not None:
        run.finish()
    
    return model
    
    
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
    if wandb is not None:
        from wandb import wandb
        # Initialisation de wandb
        run = wandb.init(wandbarg)
    
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
    test_regret(model, test_loader, weights, capacities)

#Choix des dimensions du problème
dim = 5
num_feat = 20
num_item = 30
num_data_train = 500 # Taille du dataset d'entraînement
num_data_test = 100 # Taille du dataset de test
lr = 0.001
epochs = 20

# Paramètres pour wandb, mettre à None si pas d'utilisation de wandb
wandbarg_train = {
        'entity': "hugoper-polytechnique-montr-al",
        'project': "test_imle_1",
        'name': f"train_{dim}_{num_feat}_{num_item}_{num_data_train}",
        'config': {
            "learning_rate": lr,
            "architecture": "LinearRegression", # A changer si nécéssaire
            "dataset": f"train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt",
            "epochs": epochs,
        }
}

model = run_train(dim, num_feat, num_item, num_data_train, num_data_test, 20, 0.001, True, wandbarg_train)

# Paramètres pour wandb, mettre à None si pas d'utilisation de wandb
wandbarg_test = {
        'entity': "hugoper-polytechnique-montr-al",
        'project': "test_imle_1",
        'name': f"test_{dim}_{num_feat}_{num_item}_{num_data_test}",
        'config': {
            "dataset": f"test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt",
        }
}

run_test(dim, num_feat, num_item, num_data_test, model, True, wandbarg_test)

