############# code adapté de : https://github.com/PredOpt/predopt-benchmarks/blob/main/Portfolio/Trainer/optimizer_module.py #############


import numpy as np
import torch 
from torch import nn, optim
import torch.nn.functional as F

###################################### Gurobi  Solver #########################################
import gurobipy as gp
from gurobipy import GRB
from pyepo.model.opt import optModel

class gurobi_portfolio_solver(optModel):
    '''
    Gurobi solver takes the price as parameter, return the solution of the maximizimization problem
    '''
    def __init__(self, num_item, cov, gamma, maximize = True):
        self.num_item = num_item
        self.maximize = gp.GRB.MAXIMIZE if maximize else gp.GRB.MINIMIZE
        model = gp.Model("qp")
        model.setParam('OutputFlag', 0)

        x = model.addMVar(shape= num_item, lb=0.0, vtype=GRB.CONTINUOUS, name="w")
        model.addConstr(x @ cov @ x <= gamma*np.mean(cov) , "2")
        self.model = model
        self.x = x
    
    def setObj(self, price):
        print(torch.mean(torch.abs(price)))
        self.c = price.detach().cpu().numpy()
        
    
    def _getModel(self):
        return self.model, self.x
    
    def solve(self):
        model = self.model
        x =  self.x
        model.setObjective(0, self.maximize)
        model.setObjective(self.c@x, self.maximize)
        model.optimize()

        if model.status==2:
            sol = x.x
            sol[sol < 1e-4] = 0
            return sol, None
        else:
            print(self.c)
            print(model.status)
            model.printStats()
            model.printQuality()
            raise Exception("Optimal Solution not found")   
        
    def solution_fromtorch(self, y_torch):
        if y_torch.dim()==1:
            return torch.from_numpy(self.solve( y_torch.detach().numpy())).float()
        else:
            solutions = []
            for ii in range(len(y_torch)):
                solutions.append(torch.from_numpy(self.solve( y_torch[ii].detach().numpy())).float())
            return torch.stack(solutions)