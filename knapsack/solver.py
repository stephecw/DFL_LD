import torch
import numpy as np
from numba import cuda
from pyepo.model.grb import knapsackModel
import gurobipy as gp
from joblib import Parallel, delayed

class solver_X_1D_knapsack_GPU():

    def __init__(self, weights, capacity, device):
        self.weights = weights
        self.capacity = torch.tensor([capacity], dtype=torch.int32, device=device)
        self.num_items = weights.shape[0]
        self.device = device
    
    def __call__(self, c):
        batch_size = c.shape[0]
        self.X = torch.tensor([0]*batch_size*self.num_items, dtype=torch.int32, device=self.device)
        c = c.clone().view(batch_size*self.num_items).to(self.device)
        Ldp = torch.zeros((batch_size, self.capacity.item() + 1, self.num_items + 1), dtype=torch.float32, device=self.device)
        
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
        else:
            X[idx_batch * num_items + i - 1] = 0

class solver_X_knapsack():
    """
    Solver for the knapsack problem with multiple dimensions.
    """
    def __init__(self, weights, capacities):
        self.solver = knapsackModel(weights=weights, capacity=capacities)

    def __call__(self, c_batch):
        """
        c_batch : (B, n) torch.Tensor
        -> renvoie sol_batch : (B, n) torch.Tensor
        """
        # on passe en numpy pour joblib
        c_np = c_batch
        X = np.zeros((c_np.shape[0], c_np.shape[1]), dtype=int)
        for i in range(c_np.shape[0]):
            self.solver.setObj(c_np[i])
            X[i], _ = self.solver.solve()
        return X

class solver_X_knapsack_one():
    """
    Solver for the knapsack problem with multiple dimensions.
    """
    def __init__(self, weights, capacities):
        self.solver = knapsackModel(weights=weights, capacity=capacities)

    def __call__(self,c):
        """
        c_batch : (B, n) torch.Tensor
        -> renvoie sol_batch : (B, n) torch.Tensor
        """
        # on passe en numpy pour joblib
        self.solver.setObj(c)
        X, _ = self.solver.solve()
        return X