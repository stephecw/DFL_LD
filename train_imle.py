import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import pyepo
from pyepo.metric import regret
from my_solver import solve_main_problem

# Charger les données
with open("data_knapsack.pkl", "rb") as f:
    data = pickle.load(f)

Xtrain = data["Xtrain"]
ctrain = data["ctrain"]
dataset = data["dataset"]

# Modèle d'optimisation (knapsack multi-dim)
optmodel = dataset.model

# Dataset pour pyepo
opt_dataset = pyepo.data.dataset.optDataset(optmodel, Xtrain, ctrain)
dataloader = DataLoader(opt_dataset, batch_size=32, shuffle=True)

# Modèle prédictif (régression linéaire)
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(Xtrain.shape[1], ctrain.shape[1])

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()

# Optimiseur et fonction de perte
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()

# i-MLE
imle = pyepo.func.implicitMLE(optmodel, n_samples=10, sigma=1.0, lambd=10, two_sides=False, processes=2)

# Entraînement
epochs = 20

def train(model, dataloader, optimizer, imle, epochs=20):
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
            c_hat = model(x)  # prédiction des coûts ĉ

            wp_batch = []
            for i in range(x.size(0)):
                mu_i = mu[i]
                x_opt = solve_main_problem(c_hat, mu, weights, capacities)
                wp_batch.append(x_opt)

            wp = torch.stack(wp_batch)

            # Regret = c · (w - wp)
            loss = torch.sum(c * (w - wp), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Regret (loss): {total_loss:.4f}")



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