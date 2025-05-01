import pyepo.data.dataset as dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class ImportDataset:
    """Classe pour gérer l'import de données pour l'entrainement ou le test
    """

    def __init__(self, fname, model=None, z_stats=None):
        """

        Args:
            fname (str): emplacement du dataset à importer
            model (optModel, optional): Si non None, créer un optDataset incluant model au lieu d'un TensorDataset. Defaults to None.
            z_stats ((float array (,num_feat) ou (num_data ,num_feat), float array (,num_feat) ou (num_data ,num_feat)), optional):
            paramètres de normalisation des features. Si None, une normalisation centrée réduite est effectuée. Defaults to None.
        """

        self.read_file(fname)

        Z_tensor = torch.tensor(self.Z, dtype=torch.float32)
        # Normalisation
        if z_stats is None:
            self.z_mean = Z_tensor.mean(dim=0, keepdim=True)
            self.z_std  = Z_tensor.std (dim=0, keepdim=True)
            self.z_std[self.z_std == 0] = 1.0         # évite div/0
        else:                         
            self.z_mean, self.z_std = z_stats
        Z_tensor = (Z_tensor - self.z_mean) / self.z_std 
        self.Z_tensor  = Z_tensor
        
        self.r_tensor  = torch.tensor(self.c , dtype=torch.float32)
        self.x_tensor  = torch.tensor(self.x , dtype=torch.float32)
        self.X_tensor  = torch.tensor(self.X , dtype=torch.float32)
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
        Lit le fichier de données à l'emplacement fname.
        """
        with open(fname, 'r') as f:
            lines = f.readlines()
            # Lecture des tailles
            self.num_data, self.num_feat, self.num_item, self.gamma = map((int, int, int, float), lines[0].split(","))
            
            # Lecture de cov
            self.cov = []
            for i in range(1, self.num_item + 1):
                self.cov.append(list(map(float, lines[i].split(","))))
            self.cov = np.array(self.cov)
            
            # Lecture des données des problèmes
            self.Z = []
            self.c = []
            self.x = []
            self.X = []
            self.mu = []
            for i in range(self.num_data):
                line = lines[self.num_item+1+i].split(",")
                self.Z.append(list(map(float, line[:self.num_feat])))
                self.c.append(list(map(float, line[self.num_feat : self.num_feat + self.num_item])))
                self.x.append(list(map(float, line[self.num_feat + self.num_item : self.num_feat + 2*self.num_item])))
                self.X.append(list(map(float, line[self.num_feat + 2*self.num_item : self.num_feat + 3*self.num_item])))
                self.mu.append(list(map(float, line[self.num_feat + 3*self.num_item :])))
            self.Z = np.array(self.Z)
            self.c = np.array(self.r)
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
        Retourne le DataLoader PyTorch, avec des batchs de (Z, c, x, X°, mu).
        batch_size : int : Taille des batchs.
        shuffle : bool : Si True, mélange les données.
        """
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader

