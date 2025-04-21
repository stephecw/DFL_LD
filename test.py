import torch
from torch import optim
from data_import import ImportDataset
from train_imle import LinearRegression, train, test_regret

def main(dim, num_feat, num_item, num_data_train, num_data_test, verbose=False):
    """
    Fonction principale pour charger le dataset, créer le modèle et l'entraîner.
    dim : int : Nombre de dimensions.
    num_feat : int : Nombre de features.
    num_item : int : Nombre d'items.
    num_data_train : int : Nombre de données d'entraînement.
    num_data_test : int : Nombre de données de test.
    verbose : bool : Si True, affiche des informations sur le chargement du dataset.
    """
    
    # Chargement du train dataset
    if verbose:
        print(f"Loading train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")
    try:
        train_set = ImportDataset(f"datasets/train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")
    except FileNotFoundError:
        print(f"File not found. Generating dataset with {num_data_train} data.")
        return
    # Chargement du test dataset
    if verbose:
        print(f"Loading test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt")
    try:
        test_set = ImportDataset(f"datasets/test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt")
    except FileNotFoundError:
        print(f"File not found. Generating dataset with {num_data_test} data.")
        return

    # Construction des dataloaders
    train_loader = train_set.get_dataloader()
    test_loader = test_set.get_dataloader()

    # Paramètres du modèle
    weights = train_set.get_weights(tensor=True)
    capacities = train_set.get_capacities(tensor=True)

    # Modèle, optimiseur et scheduleur
    model = LinearRegression(num_feat, num_item)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Entraînement
    if verbose:
        print("Training the model...")
    train(model, train_loader, optimizer, scheduler, weights, capacities, epochs=20)
    torch.save(model.state_dict(), f'{dim}_{num_feat}_{num_item}_{num_data_train}.pth')

    # Test
    if verbose:
        print("Testing the model...")
    test_regret(model, test_loader, weights, capacities)
    
    
#Choix des dimensions du problème
dim = 5
num_feat = 20
num_item = 30
num_data_train = 500 # Taille du dataset d'entraînement
num_data_test = 100 # Taille du dataset de test

main(dim, num_feat, num_item, num_data_train, num_data_test)