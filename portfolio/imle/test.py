from tkinter import N
import numpy as np
import torch
from data_import import ImportDataset



def diff(dataloader):
    diffs_x_X = []
    for _, _, x, X1, _ in dataloader:
        for i in range(x.size(0)):
            x_true = x[i]
            diff_x_X = torch.sum((x_true - X1[i])**2)
            diffs_x_X.append(diff_x_X.item())
    return diffs_x_X

#Choix des dimensions du problème
num_feat = 200
num_data_train = 500 # Taille du dataset d'entraînement
num_data_test = 100 # Taille du dataset de test
dim = [5]
n_item = [30]

for d in dim:
    for n in n_item:
        test_set = ImportDataset(f"datasets1000/train_{d}_{num_feat}_{n}_{num_data_train}.txt")
        test_loader = test_set.get_dataloader(batch_size=32, shuffle=False)
        diff_x_X = diff(test_loader)
        diff_x_X = np.array(diff_x_X)
        print(f"Dim {d}, n_item {n} : diff_x_X moy = {diff_x_X.mean()}, rapport = {diff_x_X.mean()/n}")