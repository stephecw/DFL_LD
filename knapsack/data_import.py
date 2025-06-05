import pyepo.data.dataset as dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class ImportDataset:
    def __init__(self, fname, model=None, z_stats=None, test=False):
        self.read_file(fname, test)

        Z_tensor = torch.tensor(self.Z, dtype=torch.float32)

        if z_stats is None:               # on calcule les stats sur CE dataset
            self.z_mean = Z_tensor.mean(dim=0, keepdim=True)
            self.z_std  = Z_tensor.std (dim=0, keepdim=True)
            self.z_std[self.z_std == 0] = 1.0         # évite div/0
        else:                             # on ré-utilise celles passées
            self.z_mean, self.z_std = z_stats
        Z_tensor = (Z_tensor - self.z_mean) / self.z_std

        self.Z_tensor  = Z_tensor
        self.c_tensor  = torch.tensor(self.c , dtype=torch.float32)
        self.x_tensor  = torch.tensor(self.x , dtype=torch.int32)
        self.X_tensor  = torch.tensor(self.X , dtype=torch.int32)
        self.mu_tensor = torch.tensor(self.mu, dtype=torch.float32)


        if model is not None:
            self.dataset = dataset.optDataset(
                model, self.Z_tensor, self.c_tensor,
                self.x_tensor, self.X_tensor, self.mu_tensor
            )
        else:
            self.dataset = TensorDataset(
                self.Z_tensor, self.c_tensor,
                self.x_tensor, self.X_tensor, self.mu_tensor
            )
    
    def read_file(self,fname,test):
        """
        Lit le fichier de données.
        """
        with open(fname, 'r') as f:
            lines = f.readlines()
            self.global_dim, self.keep ,self.num_feat, self.num_item, self.num_data = map(int, lines[0].split(","))
                
            self.capacities = []
            self.weights = []
            for i in range(self.global_dim):
                line = lines[i+1].split(",")
                self.capacities.append(int(line[0]))
                self.weights.append(list(map(int, line[1:])))
            self.capacities = np.array(self.capacities)
            self.weights = np.array(self.weights)
            
            self.obj = []
            self.Z = []
            self.c = []
            self.x = []
            self.X = []
            self.mu = []
            for i in range(self.num_data):
                line = lines[self.global_dim+1+i].split(",")
                self.obj.append(float(line[0])) 
                self.Z.append(list(map(float, line[1:1+self.num_feat])))
                self.c.append(list(map(int, line[1+self.num_feat:1+self.num_feat+self.num_item])))
                self.x.append(list(map(int, line[1+self.num_feat+self.num_item:1+self.num_feat+self.num_item+self.num_item])))
                if not test:
                    self.X.append(list(map(int, line[1+self.num_feat+self.num_item+self.num_item:1+self.num_feat+self.num_item+self.num_item+self.num_item])))
                    self.mu.append(list(map(float, line[1+self.num_feat+self.num_item+self.num_item+self.num_item:])))
            self.obj = np.array(self.obj)
            self.Z = np.array(self.Z)
            self.c = np.array(self.c)
            self.x = np.array(self.x)
            if test:
                self.X = np.zeros((self.num_data, self.num_item))
                self.mu = np.zeros((self.num_data, self.global_dim-self.keep, self.num_item))
            else: 
                self.X = np.array(self.X)
                self.mu = np.array(self.mu).reshape(self.num_data, self.global_dim-self.keep, self.num_item)

    def get_z_stats(self):
        """Retourne (mean, std) utilisés pour la normalisation."""
        return self.z_mean, self.z_std

    def get_obj(self, tensor=False):
        """
        Retourne les valeurs de l'objectif.
        tensor : bool : Si True, retourne un tenseur PyTorch.
        """
        if tensor:
            return torch.tensor(self.obj, dtype=torch.float32)
        return self.obj

    def get_sizes(self):
        """
        Retourne les tailles du dataset sous la forme (dim, num_feat, num_item, num_data).
        global_dim : int : Nombre de contraintes dans le problème primal.
        main_dim : int : Nombre de contraintes dans le sous-problème principal
        num_feat : int : Nombre de features.
        num_item : int : Nombre d'items.
        num_data : int : Nombre de données.
        """
        return self.global_dim, self.num_feat, self.num_item, self.num_data
    
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