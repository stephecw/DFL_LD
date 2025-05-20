import torch
from numba import cuda

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