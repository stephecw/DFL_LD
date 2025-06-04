import torch
import numpy as np
from numba import cuda
from pyepo.model.grb import optGrbModel
import gurobipy as gp
from gurobipy import GRB
from joblib import Parallel, delayed

class solver_partial_shortest_path(optGrbModel):

    def __init__(self, A, b):
        self.A = A
        self.b = b
        super().__init__()
    
    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        m = gp.Model("partial_shortest_path")
        # varibles
        x = m.addMVar(self.A.shape[1], name="x", vtype=GRB.BINARY)
        # sense
        m.ModelSense = GRB.MINIMIZE
        # constraints
        m.addConstr(self.A @ x == self.b)
        return m, x
    
class solver_partial_shortest_path_batch():

    def __init__(self, A, b, device):
        self.A = A
        self.b = b
        self.device = device
    
    def _solve_one(self, c_i_np):
        solver = solver_partial_shortest_path(self.A, self.b)
        solver.setObj(c_i_np)
        x_i, _ = solver.solve()
        return x_i

    def __call__(self, c_batch):
        """
        c_batch : (B, n) torch.Tensor
        
        return : sol_batch : (B, n) torch.Tensor
        """
        c_np = c_batch.detach().cpu().numpy()
        sols = Parallel(n_jobs=-1, backend="loky")(
            delayed(self._solve_one)(c_np[i]) for i in range(c_np.shape[0])
        )
        sols_np = np.stack(sols, axis=0)
        return torch.from_numpy(sols_np).to(device=self.device, dtype=c_batch.dtype)