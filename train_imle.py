import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader
import pyepo
from pyepo.metric import regret
from data_import import ImportDataset

optmodel = None ## A CHANGER

# Importer le dataset
fname = "datasets/train_5_20_30_1000.txt"
train_set = ImportDataset(fname, optmodel)

# Charger la taille du dataset
dim, num_feat, num_item, num_data = train_set.get_sizes()

# Charger les contraintes
capacities = train_set.get_capacities()
weights = train_set.get_weights()

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

train(model, train_loader, optimizer, imle, epochs)


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