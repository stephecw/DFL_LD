import torch
import pyepo.func.perturbed as pyepo_func_pert
import pyepo.func.surrogate as pyepo_func_surr

class MSE():
    def __init__(self):
        self.criterion = torch.nn.MSELoss()
    
    def __call__(self, c_hat, c, x):
        return self.criterion(c_hat, c)


class I_MLE():
    """
    Class for diff_method 
    """
    def __init__(self, solver, device, **args_imle):
        is_cpu = (device.type == "cpu")
        proc = 1 if is_cpu else 1
        self.imle = pyepo_func_pert.implicitMLE(solver, processes = proc, **args_imle)
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
        is_cpu = (device.type == "cpu")
        proc = 1 if is_cpu else 1
        self.spo = pyepo_func_surr.SPOPlus(solver, processes = proc, **args_spo)
        self.device = device
    
    def __call__(self, c_hat, c, x):
        loss = self.spo(c_hat, c, x.float(), (c*x).sum(dim=1)).to(self.device)
        return loss

class SPOPlus2():
    """
    Class for diff_method 
    """
    def __init__(self, solver, device, **args_spo):
        is_cpu = (device.type == "cpu")
        proc = 1 if is_cpu else 1
        self.spo = pyepo_func_surr.SPOPlus(solver, processes = proc, **args_spo)
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