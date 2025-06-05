import torch
import pyepo.func.perturbed as pyepo_func_pert
import pyepo.func.surrogate as pyepo_func_surr


class I_MLE():
    """
    Class for diff_method 
    """
    def __init__(self, solver, device, **args_imle):
        self.imle = pyepo_func_pert.implicitMLE(solver, **args_imle)
        self.device = device
    
    def __call__(self, c_hat, c, x):
        x_ = self.imle(c_hat).to(self.device)
        loss = torch.sum(c * (x - x_), dim=1).mean()
        return loss
    
class SPOPlus():
    """
    Class for diff_method 
    """
    def __init__(self, solver, device, **args_spo):
        self.spo = pyepo_func_surr.SPOPlus(solver, **args_spo)
        self.device = device
    
    def __call__(self, c_hat, c, x):
        loss = self.spo(-c_hat, -c, x.float(), -(c*x).sum(dim=1)).to(self.device)
        return loss
    
class Exact():
    """
    Works only for the portfolio problem, since we have a formula for the exact solution
    """
    def __init__(self, solver, device):
        self.solver = solver
        self.device = device
    
    def __call__(self, c_hat, c, x):
        x_ = self.solver(c_hat).to(self.device)
        loss = torch.sum(c * (x - x_), dim=1).mean()
        return loss