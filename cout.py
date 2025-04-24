import numpy as np
from data_import import ImportDataset

num_feat = 200
num_data_train = 500 # Taille du dataset d'entraînement
num_data_test = 100 # Taille du dataset de test
dim = 5
num_item = 50

test_set = ImportDataset(f"datasets/test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt")
train_set = ImportDataset(f"datasets/train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")

# Chargement du dataloader de test
train_loader = train_set.get_dataloader(batch_size=32, shuffle=True)
test_loader = test_set.get_dataloader(batch_size=32, shuffle=False)

total_cout = 0

for z, c, x, _, _ in test_loader:
    # z : features
    # c : capacities
    # x : labels
    # _ : weights
    # _ : indices
    z = z.numpy()
    c = c.numpy()
    x = x.numpy()

    for i in range(len(z)):
        total_cout += np.dot(c[i], x[i])

print("Mean cost:", total_cout / len(test_loader.dataset))