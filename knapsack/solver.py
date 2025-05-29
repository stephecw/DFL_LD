import torch
import numpy as np
from numba import cuda
from pyepo.model.grb import knapsackModel
import gurobipy as gp
from joblib import Parallel, delayed

class solver_X_1D_knapsack():

    def __init__(self, weights, capacity, device):
        self.weights = torch.tensor(weights, dtype=torch.int32, device=device)
        self.capacity = torch.tensor([capacity], dtype=torch.int32, device=device)
        self.num_items = weights.shape[0]
        self.device = device
    
    def __call__(self, c):
        batch_size = c.shape[0]
        self.X = torch.tensor([0]*batch_size*self.num_items, dtype=torch.int32, device=self.device)
        c = c.clone().view(batch_size*self.num_items).to(self.device)
        Ldp = torch.zeros((batch_size, self.capacity.item() + 1, self.num_items + 1), dtype=torch.int32, device=self.device)
        
        dp_knapsack_gpu_batch[batch_size, 1](self.capacity, 
                                             self.weights, 
                                             c, 
                                             self.num_items, 
                                             self.X, 
                                             Ldp)
        
        return self.X.view(batch_size, self.num_items)
        
@cuda.jit
def dp_knapsack_gpu_batch(capacity, weights, c, num_items, X, Ldp):
    idx_batch = cuda.grid(1)
    cap = capacity[0]
    for i in range(1, num_items + 1):
        for w in range(cap + 1):
            if weights[i - 1] <= w:
                Ldp[idx_batch][w][i] = max(Ldp[idx_batch][w][i - 1],
                                            Ldp[idx_batch][w - weights[i - 1]][i - 1] + c[idx_batch * num_items + i - 1])
            else:
                Ldp[idx_batch][w][i] = Ldp[idx_batch][w][i - 1]

    # Backtracking to find selected items
    w = capacity[0]
    for i in range(num_items, 0, -1):
        if Ldp[idx_batch][w][i] != Ldp[idx_batch][w][i - 1]:
            X[idx_batch * num_items + i - 1] = 1
            w -= weights[i - 1]

class solver_X_1D_knapsack_test():

    def __init__(self, weights, capacity, device):
        self.weights = torch.tensor(weights, dtype=torch.int32, device=device)
        self.capacity = torch.tensor([capacity], dtype=torch.int32, device=device)
        self.num_items = weights.shape[0]
        self.violation = None
        self.device = device
    
    def __call__(self, c):
        batch_size = c.shape[0]
        self.X = torch.tensor([0]*batch_size*self.num_items, dtype=torch.int32, device=self.device)
        c = c.clone().view(batch_size*self.num_items).to(self.device)
        Ldp = torch.zeros((batch_size, self.capacity.item() + 1, self.num_items + 1), dtype=torch.int32, device=self.device)
        
        dp_knapsack_gpu_batch[batch_size, 1](self.capacity, 
                                             self.weights, 
                                             c, 
                                             self.num_items, 
                                             self.X, 
                                             Ldp)
        
        
        
        return self.X.view(batch_size, self.num_items)
        
@cuda.jit
def dp_knapsack_gpu_batch(capacity, weights, c, num_items, X, Ldp):
    idx_batch = cuda.grid(1)
    cap = capacity[0]
    for i in range(1, num_items + 1):
        for w in range(cap + 1):
            if weights[i - 1] <= w:
                Ldp[idx_batch][w][i] = max(Ldp[idx_batch][w][i - 1],
                                            Ldp[idx_batch][w - weights[i - 1]][i - 1] + c[idx_batch * num_items + i - 1])
            else:
                Ldp[idx_batch][w][i] = Ldp[idx_batch][w][i - 1]

    # Backtracking to find selected items
    w = capacity[0]
    for i in range(num_items, 0, -1):
        if Ldp[idx_batch][w][i] != Ldp[idx_batch][w][i - 1]:
            X[idx_batch * num_items + i - 1] = 1
            w -= weights[i - 1]


class solver_X_MD_knapsack():
    """
    Solver for the knapsack problem with multiple dimensions.

    """
    def __init__(self, weights, capacities, device):
        self.weights = torch.tensor(weights, dtype=torch.int32, device=device)
        self.capacities = torch.tensor(capacities, dtype=torch.int32, device=device)
        self.num_items = weights.shape[0]
        self.device = device
    
    def _solve_one(self, c_i_np):
        solver = knapsackModel(weights=self.weights.cpu().numpy(), capacity=self.capacities.cpu().numpy())
        solver.setObj(c_i_np)
        x_i, _ = solver.solve()
        return x_i

    def __call__(self, c_batch):
        """
        c_batch : (B, n) torch.Tensor
        -> renvoie sol_batch : (B, n) torch.Tensor
        """
        # on passe en numpy pour joblib
        c_np = c_batch.detach().cpu().numpy()
        sols = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self._solve_one)(c_np[i]) for i in range(c_np.shape[0])
        )
        # retour en torch.Tensor sur l'appareil souhaité
        return torch.tensor(sols, dtype=c_batch.dtype, device=self.device)
