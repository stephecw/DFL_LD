import pickle
from tracemalloc import start
import torch
from torch import nn
from torch.utils.data import DataLoader
import pyepo
from pyepo.metric import regret
from pyepo.model.grb import knapsackModel
import gurobipy as gp
from pyepo.func import implicitMLE
from data_import import ImportDataset
from my_solver import CustomOptModel
from IMLE_perso import CustomIMLE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modèle prédictif (régression linéaire)
class LinearRegression(nn.Module):
    def __init__(self, num_feat, num_item):
        super().__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        return self.linear(x)
    
class CustomMLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU):
        super(CustomMLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Ajouter une activation sauf pour la dernière couche
                layers.append(activation())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def train(model, run, dataloader_train, dataloader_test, optimizer, scheduler, weights, capacities, epochs=20, 
          IMLE_n_samples=10, IMLE_sigma=1.0, IMLE_lambd=10, IMLE_two_sides=False, IMLE_processes=1,
          verbose=False):
    """
        Entraînement du modèle avec i-MLE classique
        model: modèle prédictif des profits
        run: wandb.run pour l'enregistrement des résultats
        dataloader: DataLoader avec (z, c, x, X1*(c), mu(c))
        optimizer: optimiseur PyTorch
        scheduler: planificateur d'apprentissage
        weights: matrice [m, n] des poids
        capacities: vecteur [m] des capacités
        epochs: nombre d’époques d'entraînement
        IMLE_n_samples: nombre d'échantillons pour i-MLE
        IMLE_sigma: sigma pour i-MLE
        IMLE_lambd: lambda pour i-MLE
        IMLE_two_sides: bool : Si True, utilise une perturbation à deux côtés
        IMLE_processes: int : Nombre de processus pour i-MLE
        verbose: bool : Si True, affiche les détails de l'entraînement
    """
    if run is not None:
        import time
        start_time = time.time()
        train_time = 0

    # multiKPModel prend en charge plusieurs contraintes
    optmodel = knapsackModel(weights=weights, capacity=capacities)
    # i-MLE avec solveur exact multi-contrainte
    imle = implicitMLE(optmodel, n_samples=IMLE_n_samples, sigma=IMLE_sigma, lambd=IMLE_lambd,
                       two_sides=IMLE_two_sides, processes=IMLE_processes)

    for epoch in range(epochs):
        if run is not None:
            epoch_start_time = time.time()
        model.train()
        total_loss = 0

        for z, c, x, X1, mu in dataloader_train:
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            cp = model(z)
            xp = imle(cp).to(device)

            # Regret = c · (w - wp)
            loss = torch.sum(c * (x - xp), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        if scheduler is not None:
                scheduler.step()
        
        mean_loss = total_loss / len(dataloader_train)
        if run is not None:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            train_time += epoch_duration
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration, "train_time": train_time})
        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")
            
        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                total_regret = 0
                total_count = 0
                for z, c, x, _, _ in dataloader_test:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # prédiction des coûts [batch, n]
                    batch_regrets = []

                    for i in range(z.size(0)):
                        model_i = knapsackModel(weights=weights, capacity=capacities)
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        model_i.setObj(c_numpy)
                        x_hat_np, _ = model_i.solve()

                        x_true = x[i].to(dtype=torch.float32, device=device)
                        c_true = c[i].to(dtype=torch.float32, device=device)

                        x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32,
                                            device=device)

                        regret = torch.dot(c_true, x_true - x_hat_tensor)
                        batch_regrets.append(regret)

                    batch_regrets = torch.stack(batch_regrets)
                    total_regret += batch_regrets.sum().item()
                    total_count += z.size(0)
                
            mean_regret = total_regret / total_count
            if run is not None:
                # Enregistrement des résultats dans wandb
                run.log({"epoch": epoch, "regret": mean_regret})
            if verbose:
                print(f"Eval Epoch {epoch+1} | Regret moyen : {mean_regret:.4f}")
            
    if run is not None:   
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})


def train_LD(model, run, dataloader_train, dataloader_test, optimizer, scheduler, weights, capacities, epochs=20, 
          IMLE_n_samples=10, IMLE_sigma=1.0, IMLE_lambd=10, IMLE_two_sides=False, IMLE_processes=1,
          verbose=False):
    """
        Entraînement du modèle avec la décomposition lagrangienne
        model: modèle prédictif des profits
        run: wandb.run pour l'enregistrement des résultats
        dataloader: DataLoader avec (z, c, x, X1*(c), mu(c))
        optimizer: optimiseur PyTorch
        scheduler: planificateur d'apprentissage
        weights: matrice [m, n] des poids
        capacities: vecteur [m] des capacités
        epochs: nombre d’époques d'entraînement
        IMLE_n_samples: nombre d'échantillons pour i-MLE
        IMLE_sigma: sigma pour i-MLE
        IMLE_lambd: lambda pour i-MLE
        IMLE_two_sides: bool : Si True, utilise une perturbation à deux côtés
        IMLE_processes: int : Nombre de processus pour i-MLE
        verbose: bool : Si True, affiche les détails de l'entraînement
    """
    if run is not None:
        import time
        start_time = time.time()
        train_time = 0
    
    # Créer un solveur i-MLE avec les mu du batch
    solver = knapsackModel(weights[0].unsqueeze(0), capacities[0].unsqueeze(0))
    imle = CustomIMLE(solver, n_samples=IMLE_n_samples, sigma=IMLE_sigma, lambd=IMLE_lambd,
                       two_sides=IMLE_two_sides, processes=IMLE_processes)

    for epoch in range(epochs):
        if run is not None:
            epoch_start_time = time.time()
        model.train()
        total_loss = 0

        for z, c, x, X1, mu in dataloader_train: # x vrai solution et X1 solution avec mu(c)
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            c_hat = model(z)  # prédiction des profits ĉ


            # Résolution avec i-MLE
            X1p = imle(c_hat, mu).to(device)   # x̂ obtenu avec solve_main_problem

            # (c + sum mu_i for i ≥ 2) · (w - x̂)
            mu_sum = mu.sum(dim=1)  # shape [batch, n]
            profit_modified = c + mu_sum     # shape [batch, n]
            loss = torch.sum(profit_modified * (X1 - X1p), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            total_loss += loss.item()

        if scheduler is not None:
                scheduler.step()

        mean_loss = total_loss / len(dataloader_train)
        if run is not None:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            train_time += epoch_duration
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration, "train_time": train_time})
        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")

        
        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                total_regret = 0
                total_count = 0
                for z, c, x, _, _ in dataloader_test:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # prédiction des coûts [batch, n]
                    batch_regrets = []

                    for i in range(z.size(0)):
                        model_i = knapsackModel(weights=weights, capacity=capacities)
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        model_i.setObj(c_numpy)
                        x_hat_np, _ = model_i.solve()

                        x_true = x[i].to(dtype=torch.float32, device=device)
                        c_true = c[i].to(dtype=torch.float32, device=device)

                        x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32,
                                        device=device)

                        regret = torch.dot(c_true, x_true - x_hat_tensor)
                        batch_regrets.append(regret)

                    batch_regrets = torch.stack(batch_regrets)
                    total_regret += batch_regrets.sum().item()
                    total_count += z.size(0)
                
            mean_regret = total_regret / total_count
            if run is not None:
                # Enregistrement des résultats dans wandb
                run.log({"epoch": epoch, "regret": mean_regret})
            if verbose:
                print(f"Eval Epoch {epoch+1} | Regret moyen : {mean_regret:.4f}")
            
    if run is not None:   
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})

def test_regret(model, dataloader, weights, capacities, run = None,verbose=False):
    """
    Évaluation du modèle avec résolution exacte : regret = c · (x - x̂)
    model: modèle prédictif des profits
    run: wandb.run pour l'enregistrement des résultats
    dataloader: DataLoader avec (z, c, x, X1*(c), mu(c))
    weights: matrice [m, n] des poids
    capacities: vecteur [m] des capacités
    verbose: bool : Si True, affiche les détails de l'évaluation
    """
    model.eval()
    total_regret = 0
    total_count = 0

    with torch.no_grad():
        model.eval()
        total_regret = 0
        total_count = 0
        for z, c, x, _, _ in dataloader:
            z, c, x = [t.to(device) for t in (z, c, x)]
            c_hat = model(z)  # prédiction des coûts [batch, n]
            batch_regrets = []

            for i in range(z.size(0)):
                model_i = knapsackModel(weights=weights, capacity=capacities)
                c_numpy = c_hat[i].detach().cpu().numpy()
                model_i.setObj(c_numpy)
                x_hat_np, _ = model_i.solve()

                x_true = x[i].to(dtype=torch.float32, device=device)
                c_true = c[i].to(dtype=torch.float32, device=device)

                x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32,
                                        device=device)

                regret = torch.dot(c_true, x_true - x_hat_tensor)
                batch_regrets.append(regret)

            batch_regrets = torch.stack(batch_regrets)
            total_regret += batch_regrets.sum().item()
            total_count += z.size(0)
                
    mean_regret = total_regret / total_count
    if run is not None:
        # Enregistrement des résultats dans wandb
        run.log({"regret": mean_regret})
    if verbose:
        print(f"\n Regret moyen exact (c · (x - x̂)) : {mean_regret:.4f}")
    return mean_regret

