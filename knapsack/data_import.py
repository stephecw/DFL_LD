from ctypes import pointer
import pyepo.data.dataset as dataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

class ImportDataset:
    def __init__(self, fname, device, normalize_feat=True):
        self.device = device
        self.read_file(fname)

        Z_tensor = torch.tensor(self.Z, dtype=torch.float32)
        self.z_mean = Z_tensor.mean(dim=0, keepdim=True)
        self.z_std  = Z_tensor.std(dim=0, keepdim=True)
        self.z_std[self.z_std == 0] = 1.0 
        
        if normalize_feat: # Normalize features              
            Z_tensor = (Z_tensor - self.z_mean) / self.z_std

        self.Z_tensor  = Z_tensor.to(self.device)
        self.c_tensor  = torch.tensor(self.c , dtype=torch.float32, device=self.device)
        self.x_tensor  = torch.tensor(self.x , dtype=torch.int32, device=self.device)
        self.X_tensor  = torch.tensor(self.X , dtype=torch.int32, device=self.device)
        self.mu_tensor = torch.tensor(self.mu, dtype=torch.float32, device=self.device)
    
    def read_file(self,fname):
        """Extract data from a given file.

        Args:
            fname (str): file name to get data from.
        """
        with open(fname, 'r') as f:
            lines = f.readlines()
            line = lines[0].split(",")
            self.main = None
            if len(line) == 5:
                train = False
                self.dim, self.num_feat, self.num_items, self.num_data, self.seed = map(int, line)
            else:
                train = True
                self.dim, self.main ,self.num_feat, self.num_items, self.num_data, self.seed = map(int, line)
                               
            self.capacities = []
            self.weights = []
            for line in lines[1:self.dim+1]:
                line = line.split(",")
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
            for line in lines[self.dim+1:]:
                line = line.split(",")
                self.obj.append(float(line[0])) 
                self.Z.append(list(map(float, line[1:1+self.num_feat])))
                self.c.append(list(map(float, line[1+self.num_feat:1+self.num_feat+self.num_items])))
                self.x.append(list(map(int, line[1+self.num_feat+self.num_items:1+self.num_feat+2*self.num_items])))
                if train:
                    ptr = 1 + self.num_feat + 2*self.num_items
                    if self.main == -1:
                        self.X.append(list(map(int, line[ptr:ptr+self.num_items*self.dim])))
                        self.mu.append(list(map(float, line[ptr+self.num_items*self.dim:])))
                    else:
                        self.X.append(list(map(int, line[ptr:ptr+self.num_items])))
                        self.mu.append(list(map(float, line[ptr+self.num_items:])))

            self.obj = np.array(self.obj)
            self.Z = np.array(self.Z)
            self.c = np.array(self.c)
            self.x = np.array(self.x)
            if train: 
                if self.main == -1:
                    self.X = np.reshape(np.array(self.X), (self.num_data, self.dim, self.num_items))
                    self.mu = np.reshape(np.array(self.mu), (self.num_data, self.dim, self.dim-1,self.num_items))
                else:
                    self.X = np.reshape(np.array(self.X), (self.num_data, 1, self.num_items))
                    self.mu = np.reshape(np.array(self.mu), (self.num_data, 1, self.dim-1, self.num_items))
            else:
                self.X = np.zeros((self.num_data, 1, 1))
                self.mu = np.zeros((self.num_data, 1, 1))

    def get_z_stats(self):
        """Get the mean and standard deviation of the features (before normalization if applicable).

        Returns:
            (float, float): (mean, std).
        """
        return self.z_mean, self.z_std

    def get_obj(self, tensor=False, device=torch.device("cpu")):
        """ Get the optimal values (of objective function) of each problem instances in the dataset. 

        Args:
            tensor (bool, optional): Whether to return a PyTorch tensor. Defaults to False.
            device (torch.device, optional): Device on which the PyTorch tensor is created. Defaults to 'cpu'.

        Returns:
            numpy.ndarray or torch.Tensor shape(num_data, num_items): optimal values.
        """
        if tensor:
            return torch.tensor(self.obj, dtype=torch.float32, device=device)
        return self.obj

    def get_seed(self):
        """ Get the random seed used during dataset generation.

        Returns:
            int: seed.
        """
        return self.seed

    def get_sizes(self):
        """Get various informations about dataset shape.

        Returns: tuple of int
            int: number of constraints
            int: index of main problem (None if not applicable)
            int: feature vector size
            int: number of items
            int: number of instances
        """
        return self.dim, self.main, self.num_feat, self.num_items, self.num_data
    
    def get_capacities(self, tensor=False, device=torch.device("cpu")):
        """ Get capacities of the constraints in the dataset
        Args:
            tensor (bool, optional): Whether to return a PyTorch tensor. Defaults to False.
            device (torch.device, optional): Device on which the PyTorch tensor is created. Defaults to 'cpu'.

        Returns:
            numpy.ndarray or torch.Tensor shape(dim,): capacities.
        """
        if tensor:
            return torch.tensor(self.capacities, dtype=torch.int32, device=device)
        return self.capacities
        
    def get_weights(self, tensor=False, device=torch.device("cpu")):
        """ Get weights of the constraints in the dataset
        Args:
            tensor (bool, optional): Whether to return a PyTorch tensor. Defaults to False.
            device (torch.device, optional): Device on which the PyTorch tensor is created. Defaults to 'cpu'.

        Returns:
            numpy.ndarray or torch.Tensor shape(dim, num_items): capacities.
        """
        if tensor:
            return torch.tensor(self.weights, dtype=torch.int32, device=device)
        return self.weights
    
    def get_device(self):
        """Get data device

        Returns:
            torch.device: device
        """
        return self.device
    
    def get_dataset(self):
        """Build dataset object from the data
        Shape : feature Z ; cost c ; optimal solution x* ; dual optimal solution X* ; lagr. mult. mu
        
        Returns:
            torch.utils.data.TensorDataset: dataset
        """
        self.dataset = TensorDataset(
            self.Z_tensor, self.c_tensor,
            self.x_tensor, self.X_tensor, self.mu_tensor
        )
        return self.dataset
    
    def get_dataloader(self, batch_size=32, shuffle=True):
        """Build a dataloader from the dataset

        Args:
            batch_size (int, optional): Batch size of the dataloader. Defaults to 32.
            shuffle (bool, optional): Whether to suffle data. Defaults to True.

        Returns:
            torch.utils.data.DataLoader: dataloader
        """
        self.get_dataset()
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader