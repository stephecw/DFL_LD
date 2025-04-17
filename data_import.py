import pyepo
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def read_file(fname):
    """
    Lit le fichier de données et renvoie les informations sous forme de dictionnaire.
    """
    num_data = 1000 # Taille du dataset
    num_feat = 20 # Nombre de features en entrée du NN
    num_item = 30 # Nombre d'items
    dim = 5 # Nombre de contraintes
    with open(fname, 'r') as f:
        lines = f.readlines()
        dim, num_feat, num_item, num_data = map(int, lines[0].split(","))
        capacities = []
        weights = []
        for i in range(dim):
            line = lines[i+1].split(",")
            capacities.append(int(line[0]))
            weights.append(list(map(int, line[1:])))
        capacities = np.array(capacities)
        weights = np.array(weights)
        Z = []
        c = []
        for i in range(num_data):
            line = lines[dim+1+i].split(",")
            Z.append(list(map(float, line[:num_feat])))
            c.append(list(map(int, line[num_feat:])))
        Z = np.array(Z)
        c = np.array(c)
    return {
        'dim': dim,
        'num_feat': num_feat,
        'num_item': num_item,
        'num_data': num_data,
        'capacities': capacities,
        'weights': weights,
        'Z': Z,
        'c': c
    }

def data_to_tensor(data):
    """
    Convertit les données en tenseurs PyTorch.
    """
    capacities = torch.tensor(data['capacities'], dtype=torch.int32)
    weights = torch.tensor(data['weights'], dtype=torch.int32)
    Z = torch.tensor(data['Z'], dtype=torch.float32)
    c = torch.tensor(data['c'], dtype=torch.int32)
    return capacities, weights, Z, c

def create_dataloader(z, c, batch_size=32, shuffle=True):
    """
    Crée un DataLoader à partir des tenseurs Z et c.
    """
    dataset = TensorDataset(Z, c)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

fname = "datasets/train_5_20_30_1000.txt"
data = read_file(fname)
capacities, weights, Z, c = data_to_tensor(data)
train_loader = create_dataloader(Z, c)

