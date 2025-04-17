import torch
from torch import optim
from data_import import ImportDataset
from train_imle import LinearRegression, train, test_regret

# Chargement du dataset
fname_train = "datasets/train_5_20_30_1000.txt"
fname_test = "datasets/test_5_20_30_100.txt"

train_set = ImportDataset(fname_train)
test_set = ImportDataset(fname_test)

train_loader = train_set.get_dataloader()
test_loader = test_set.get_dataloader()

# Paramètres du modèle
dim, num_feat, num_item, _ = train_set.get_sizes()
weights = train_set.get_weights(tensor=True)
capacities = train_set.get_capacities(tensor=True)

# Modèle
model = LinearRegression(num_feat, num_item)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Entraînement
train(model, train_loader, optimizer, weights, capacities, epochs=20)

# Test
test_regret(model, test_loader, weights, capacities)