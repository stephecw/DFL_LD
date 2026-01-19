import pyepo.data.dataset as dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class ImportDataset:
    """Class to handle dataset import for training or testing."""

    def __init__(self, fname, model=None, z_stats=None):
        """

        Args:
            fname (str): path to the dataset to import
            model (optModel, optional): If not None, create an optDataset including model instead of a TensorDataset. Defaults to None.
            z_stats ((float array (,num_feat) ou (num_data ,num_feat), float array (,num_feat) ou (num_data ,num_feat)), optional):
            feature normalization parameters. If None, standardization (zero mean / unit variance) is applied. Defaults to None.
        """

        self.read_file(fname)

        Z_tensor = torch.tensor(self.Z, dtype=torch.float32)
        # Normalization
        if z_stats is None:
            self.z_mean = Z_tensor.mean(dim=0, keepdim=True)
            self.z_std  = Z_tensor.std (dim=0, keepdim=True)
            self.z_std[self.z_std == 0] = 1.0         # avoid div/0
        else:                         
            self.z_mean, self.z_std = z_stats
        Z_tensor = (Z_tensor - self.z_mean) / self.z_std 
        self.Z_tensor  = Z_tensor
        
        self.c_tensor  = torch.tensor(self.c , dtype=torch.float32)
        self.x_tensor  = torch.tensor(self.x , dtype=torch.float32)
        self.X_tensor  = torch.tensor(self.X , dtype=torch.float32).unsqueeze(1)
        self.mu_tensor = torch.tensor(self.mu, dtype=torch.float32).unsqueeze(1).unsqueeze(1)


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
    
    def read_file(self,fname):
        """
        Reads the dataset file located at fname.
        """
        with open(fname, 'r') as f:
            lines = f.readlines()
            # Read sizes
            self.num_data, self.num_feat, self.num_item, self.deg, self.gamma = map(float, lines[0].split(","))

            self.num_data = int(self.num_data)
            self.num_feat = int(self.num_feat)
            self.num_item = int(self.num_item)
            
            # Read covariance
            self.cov = []
            for i in range(1, self.num_item + 1):
                self.cov.append(list(map(float, lines[i].split(","))))
            self.cov = np.array(self.cov)
            
            # Read problem instances
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
            self.c = np.array(self.c)
            self.x = np.array(self.x)
            self.X = np.array(self.X)
            self.mu = np.array(self.mu)

    def get_z_stats(self):
        """Returns (mean, std) used for normalization."""
        return self.z_mean, self.z_std

    def get_sizes(self):
        """
        Returns dataset sizes as (num_data, num_feat, num_item).
        num_data : int : number of instances.
        num_feat : int : number of features.
        num_item : int : number of items.
        """
        return self.num_data, self.num_feat, self.num_item
        
    def get_gamma(self):
        """
        Returns the gamma parameter of the portfolio problem.
        """
        return self.gamma
        
    def get_cov(self, tensor=False):
        """
        Returns the covariance matrix of the portfolio problem.
        tensor : bool : If True, returns a PyTorch tensor.
        """
        if tensor:
            return torch.tensor(self.cov, dtype=torch.float32, requires_grad=False)
        return self.cov
    
    def get_dataset(self):
        """
        Returns the PyEPO dataset.
        """
        return self.dataset
    
    def get_dataloader(self, batch_size=32, shuffle=True):
        """
        Returns the PyTorch DataLoader, yielding batches of (Z, c, x, X°, mu).
        batch_size : int : batch size.
        shuffle : bool : If True, shuffles the data.
        """
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader
    def get_mu(self):
        """
        Returns the mu vector for the portfolio problem.
        """
        return self.mu_tensor
    def get_deg(self):
        """
        Returns the polynomial degree used to generate the data.
        """
        return self.deg
