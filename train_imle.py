import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import pyepo
from pyepo.metric import regret
from pyepo.model.grb import knapsackModel
import gurobipy as gp
from pyepo.func import implicitMLE
from data_import import ImportDataset
from my_solver import get_imle_solver_with_mu

# Modèle prédictif (régression linéaire)
class LinearRegression(nn.Module):
    def __init__(self, num_feat, num_item):
        super().__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        return self.linear(x)

def train(model, run, dataloader, optimizer, scheduler, weights, capacities, epochs=20, verbose=False):
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
        verbose: bool : Si True, affiche les détails de l'entraînement
    """

    m, n = weights.shape

    # multiKPModel prend en charge plusieurs contraintes
    optmodel = knapsackModel(weights=weights, capacity=capacities)

    # i-MLE avec solveur exact multi-contrainte
    imle = implicitMLE(optmodel, n_samples=10, sigma=1.0, lambd=10, two_sides=False, processes=2)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for z, c, x, X1, mu in dataloader:
            cp = model(z)
            xp = imle(cp)

            # Regret = c · (w - wp)
            loss = torch.sum(c * (x - xp), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        
        mean_loss = total_loss / len(dataloader)
        if run is not None:
            # Enregistrement des résultats dans wandb
            run.log({"loss": mean_loss})
        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")


def train_LD(model, run, dataloader, optimizer, scheduler, weights, capacities, epochs=20, verbose=False):
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
        verbose: bool : Si True, affiche les détails de l'entraînement
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for z, c, x, X1, mu in dataloader: # x vrai solution et X1 solution avec mu(c)
            c_hat = model(z)  # prédiction des profits ĉ

            # Créer un solveur i-MLE avec les mu du batch
            solver = get_imle_solver_with_mu(weights, capacities, mu)
            imle = pyepo.func.implicitMLE(solver, n_samples=10, sigma=1.0, lambd=10)

            # Résolution avec i-MLE
            X1p = imle(c_hat)  # x̂ obtenu avec solve_main_problem

            # (c + sum mu_i for i ≥ 2) · (w - x̂)
            mu_sum = mu.sum(dim=1)  # shape [batch, n]
            profit_modified = c + mu_sum     # shape [batch, n]
            loss = torch.sum(profit_modified * (X1 - X1p), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        mean_loss = total_loss / len(dataloader)
        if run is not None:
            # Enregistrement des résultats dans wandb
            run.log({"loss": mean_loss})
        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")


def test_regret(model, run, dataloader, weights, capacities, verbose=False):
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

    m, n = weights.shape

    with torch.no_grad():
        for z, c, x, _, _ in dataloader:
            c_hat = model(z)  # prédiction des coûts [batch, n]
            batch_regrets = []

            for i in range(z.size(0)):
                model_i = knapsackModel(weights=weights, capacity=capacities)
                c_numpy = c_hat[i].detach().cpu().numpy()
                model_i.setObj(c_numpy)
                x_hat, _ = model_i.solve()

                x_true = x[i]
                c_true = c[i]
                x_hat_tensor = torch.tensor(x_hat, dtype=torch.float32)

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


