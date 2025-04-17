import pyepo
import pyepo.data.dataset as dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class ImportDataset:
    def __init__(self, fname, model=None):
        self.read_file(fname)
        
        self.Z_tensor = torch.tensor(self.Z, dtype=torch.float32)
        self.c_tensor = torch.tensor(self.c, dtype=torch.int32)
        self.x_tensor = torch.tensor(self.x, dtype=torch.int32)
        self.X_tensor = torch.tensor(self.X, dtype=torch.int32)
        self.mu_tensor = torch.tensor(self.mu, dtype=torch.float32)
        if model is not None:
            self.model = model
            self.dataset = dataset.optDataset(self.model, self.Z_tensor, self.c_tensor, self.x_tensor, self.X_tensor, self.mu_tensor)
        self.dataset = TensorDataset(self.Z_tensor, self.c_tensor, self.x_tensor, self.X_tensor, self.mu_tensor)
    
    def read_file(self,fname):
        """
        Lit le fichier de données.
        """
        with open(fname, 'r') as f:
            lines = f.readlines()
            self.dim, self.num_feat, self.num_item, self.num_data = map(int, lines[0].split(","))
            self.capacities = []
            self.weights = []
            for i in range(self.dim):
                line = lines[i+1].split(",")
                self.capacities.append(int(line[0]))
                self.weights.append(list(map(int, line[1:])))
            self.capacities = np.array(self.capacities)
            self.weights = np.array(self.weights)
            self.Z = []
            self.c = []
            self.x = []
            self.X = []
            self.mu = []
            for i in range(self.num_data):
                line = lines[self.dim+1+i].split(",")
                self.Z.append(list(map(float, line[:self.num_feat])))
                self.c.append(list(map(int, line[self.num_feat:self.num_feat+self.num_item])))
                self.x.append(list(map(int, line[self.num_feat+self.num_item:self.num_feat+self.num_item+self.num_item])))
                self.X.append(list(map(int, line[self.num_feat+self.num_item+self.num_item:self.num_feat+self.num_item+self.num_item+self.num_item])))
                self.mu.append(list(map(float, line[self.num_feat+self.num_item+self.num_item+self.num_item:])))
            self.Z = np.array(self.Z)
            self.c = np.array(self.c)
            self.x = np.array(self.x)
            self.X = np.array(self.X)
            self.mu = np.array(self.mu).reshape(self.num_data, self.dim-1, self.num_item)

    def get_sizes(self):
        """
        Retourne les tailles du dataset.
        """
        return self.dim, self.num_feat, self.num_item, self.num_data
    def get_capacities(self, tensor=False):
        """
        Retourne les capacités du dataset.
        tensor : bool : Si True, retourne un tenseur PyTorch.
        """
        if tensor:
            return torch.tensor(self.capacities, dtype=torch.int32)
        return self.capacities
        
    def get_weights(self, tensor=False):
        """
        Retourne les poids du dataset.
        tensor : bool : Si True, retourne un tenseur PyTorch.
        """
        if tensor:
            return torch.tensor(self.weights, dtype=torch.int32)
        return self.weights
    
    def get_dataset(self):
        """
        Retourne le dataset PyEPO.
        """
        return self.dataset
    
    def get_dataloader(self, batch_size=32, shuffle=True):
        """
        Retourne le DataLoader PyTorch, avec des batchs de (Z, c, X°, [mu_2, mu_3, ...]).
        batch_size : int : Taille du batch.
        shuffle : bool : Si True, mélange les données.
        """
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

