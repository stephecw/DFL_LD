import pyepo.data.dataset as dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class ImportDataset:

    def __init__(self, fname, model=None, z_stats=None):

        self.read_file(fname)

        Z_tensor = torch.tensor(self.Z, dtype=torch.float32)

        if z_stats is None:               # on calcule les stats sur CE dataset
            self.z_mean = Z_tensor.mean(dim=0, keepdim=True)
            self.z_std  = Z_tensor.std (dim=0, keepdim=True)
            self.z_std[self.z_std == 0] = 1.0         # évite div/0
        else:                             # on ré-utilise celles passées
            self.z_mean, self.z_std = z_stats
        Z_tensor = (Z_tensor - self.z_mean) / self.z_std # Normalisation

        self.Z_tensor  = Z_tensor
        self.r_tensor  = torch.tensor(self.r , dtype=torch.float32)
        self.x_tensor  = torch.tensor(self.x , dtype=torch.float32)
        self.X_tensor  = torch.tensor(self.X , dtype=torch.int32)
        self.mu_tensor = torch.tensor(self.mu, dtype=torch.float32)


        if model is not None:
            self.dataset = dataset.optDataset(
                model, self.Z_tensor, self.r_tensor,
                self.x_tensor, self.X_tensor, self.mu_tensor
            )
        else:
            self.dataset = TensorDataset(
                self.Z_tensor, self.r_tensor,
                self.x_tensor, self.X_tensor, self.mu_tensor
            )
    
    def read_file(self,fname):
        """
        Lit le fichier de données.
        """
        with open(fname, 'r') as f:
            lines = f.readlines()
            self.num_data, self.num_feat, self.num_item, self.gamma = map((int, int, int, float), lines[0].split(","))
            self.cov = []
            for i in range(self.num_item):
                self.cov.append(list(map(float, lines[i+1].split(","))))
            self.cov = np.array(self.cov)
            self.Z = []
            self.r = []
            self.x = []
            self.X = []
            self.mu = []
            for i in range(self.num_data):
                line = lines[self.dim+1+i].split(",")
                self.Z.append(list(map(float, line[:self.num_feat])))
                self.r.append(list(map(float, line[self.num_feat : self.num_feat + self.num_item])))
                self.x.append(list(map(float, line[self.num_feat + self.num_item : self.num_feat + 2*self.num_item])))
                self.X.append(list(map(int, line[self.num_feat + 2*self.num_item : self.num_feat + 3*self.num_item])))
                self.mu.append(list(map(float, line[self.num_feat + 3*self.num_item :])))
            self.Z = np.array(self.Z)
            self.r = np.array(self.r)
            self.x = np.array(self.x)
            self.X = np.array(self.X)
            self.mu = np.array(self.mu)

    def get_z_stats(self):
        """Retourne (mean, std) utilisés pour la normalisation."""
        return self.z_mean, self.z_std

    def get_sizes(self):
        """
        Retourne les tailles du dataset sous la forme (dim, num_feat, num_item, num_data).
        num_data : int : Nombre de données.
        num_feat : int : Nombre de features.
        num_item : int : Nombre d'items.
        """
        return self.num_data, self.num_feat, self.num_item
    
    def get_gamma(self, tensor=False):
        """
        Retourne le paramètre gamma du problème de portfolio.
        """
        return self.gamma
        
    def get_cov(self, tensor=False):
        """
        Retourne la matrice de covariance du problème de portfolio.
        tensor : bool : Si True, retourne un tenseur PyTorch.
        """
        if tensor:
            return torch.tensor(self.cov, dtype=torch.float32)
        return self.cov
    
    def get_dataset(self):
        """
        Retourne le dataset PyEPO.
        """
        return self.dataset
    
    def get_dataloader(self, batch_size=32, shuffle=True):
        """
        Retourne le DataLoader PyTorch, avec des batchs de (Z, r, x, X°, mu).
        batch_size : int : Taille des batchs.
        shuffle : bool : Si True, mélange les données.
        """
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

