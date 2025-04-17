import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import pyepo
from pyepo.metric import regret
from pyepo.model.grb import multiKPModel
from pyepo.func import implicitMLE
from data_import import ImportDataset
from my_solver import get_imle_solver_with_mu

# Importer le dataset
fname = "datasets/train_5_20_30_1000.txt"
train_set = ImportDataset(fname)

# Charger la taille du dataset
dim, num_feat, num_item, num_data = train_set.get_sizes()



# Charger le dataloader d'entrainement
train_loader = train_set.get_dataloader()

# Modèle prédictif (régression linéaire)
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# Optimiseur et fonction de perte
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()


# Entraînement
epochs = 20

def train(model, dataloader, optimizer, weights, capacities, epochs=20):

    m, n = weights.shape

    # multiKPModel prend en charge plusieurs contraintes
    optmodel = multiKPModel(n=n, m=m, budget=capacities, weight=weights)

    # i-MLE avec solveur exact multi-contrainte
    imle = implicitMLE(optmodel, n_samples=10, sigma=1.0, lambd=10, two_sides=False, processes=2)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, c, w, z in dataloader:
            cp = model(x)
            wp = imle(cp)

            # Regret = c · (w - wp)
            loss = torch.sum(c * (w - wp), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Regret (loss): {total_loss:.4f}")


def train_LD(model, dataloader, optimizer, weights, capacities, epochs=20):
    """
        Entraînement du modèle avec la décomposition lagrangienne
        dataloader: DataLoader avec (x, c, X1*(c), mu(c))
        optimizer: optimiseur PyTorch
        weights: matrice [m, n] des poids
        capacities: vecteur [m] des capacités
        epochs: nombre d’époques d'entraînement
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, c, w, mu in dataloader:
            c_hat = model(x)  # prédiction des profits ĉ

            # Créer un solveur i-MLE avec les mu du batch
            solver = get_imle_solver_with_mu(weights, capacities, mu)
            imle = pyepo.func.implicitMLE(solver, n_samples=10, sigma=1.0, lambd=10)

            # Résolution avec i-MLE
            wp = imle(c_hat)  # x̂ obtenu avec solve_main_problem

            # (c + sum mu_i for i ≥ 2) · (w - x̂)
            mu_sum = mu.sum(dim=1)  # shape [batch, n]
            profit_modified = c + mu_sum     # shape [batch, n]
            loss = torch.sum(profit_modified * (w - wp), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss = (c + ∑μ)·(w - ŵ) : {total_loss:.4f}")



train(model, dataloader, optimizer, imle, epochs)


# Évaluation
model.eval()
with torch.no_grad():
    Xtest = torch.tensor(data["Xtest"], dtype=torch.float)
    ctest = torch.tensor(data["ctest"], dtype=torch.float)

    # Prédiction des coûts
    c_pred = model(Xtest)

    # Calcul du regret
    reg = regret(c_pred, ctest, dataset.sol)
    print(f"\nRegret moyen sur le test : {reg.mean():.4f}")