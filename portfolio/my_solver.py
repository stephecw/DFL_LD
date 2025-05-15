import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pyepo.model.opt import optModel
from scipy.optimize import minimize

class Solveur_lin(optModel):
    def __init__(self, num_item, maximize = True):
        self.num_item = num_item
        self.maximize = maximize
        super().__init__()

    def setObj(self, r):
        self.c = r

    def _getModel(self):
        """
        Juste pour satisfaire l’interface requise par optModel.  
        """
        m = gp.Model()               # nécessite « import gurobipy as gp »
        x = [m.addVar(vtype=gp.GRB.BINARY, name=f"x{i}")
             for i in range(self.num_item)]
        m.update()
        return m, x  

    def solve(self):
        best_idx = torch.argmax(self.c) if self.maximize else torch.argmin(self.c)
        sol = torch.zeros(self.num_item, dtype=torch.float32, device=self.c.device)
        sol[best_idx] = 1
        return sol, torch.dot(self.c, sol)
    
class Solveur_quad(optModel):
    def __init__(self,  n_stocks, cov, gamma, maximize=True):
        self.n_stocks = n_stocks
        self.cov = cov
        self.gamma =  gamma
        self.maximize = maximize

        self.qp_model = gp.Model("qp")
        self.qp_model.setParam('OutputFlag', 0)

        self.qp_w = self.qp_model.addMVar(shape= n_stocks, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

        self.qp_model.addConstr(self.qp_w @ self.cov @ self.qp_w <= self.gamma*np.mean(self.cov) , "2")

        super().__init__()

    def _getModel(self):
        return self.qp_model, self.qp_w
    
    def setObj(self, r):
        self.c = r
        c_np   = self.c.detach().cpu().numpy()
        sense  = GRB.MAXIMIZE if self.maximize else GRB.MINIMIZE
        self.qp_model.setObjective(0, sense)
        self.qp_model.setObjective(c_np @ self.qp_w, sense)
    
    def solve(self):
        self.qp_model.optimize()
        if self.qp_model.status != GRB.OPTIMAL:
            print("solution pas optimale | statut : ", self.qp_model.status, "\n")
            #print(self.c.detach().cpu().numpy())
            #raise RuntimeError("Gurobi n’a pas trouvé d’optimum.")
        sol_np = self.qp_w.X.copy()
        sol_np[sol_np<1e-6] = 0      # nettoyage numérique
        obj_val = self.qp_model.ObjVal
        sol_t = torch.tensor(sol_np, dtype=self.c.dtype,
                                        device=self.c.device)
        return sol_t, obj_val 

    
class gb_portfolio_solver(optModel):
    '''
    Gurobi solver takes the price as parameter, return the solution of the maximizimization problem
    '''
    def __init__(self,  n_stocks, cov, gamma, maximize=True):
        self.n_stocks = n_stocks
        self.cov = cov
        self.gamma =  gamma
        self.maximize = maximize

        self.qp_model = gp.Model("qp")
        self.qp_model.setParam('OutputFlag', 0)

        self.qp_w = self.qp_model.addMVar(shape= n_stocks, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

        self.qp_model.addConstr(self.qp_w.sum() <= 1, "1")
        ### Original Model invoves inequality, We once tested  with Equality
        # model.addConstr(sum(x) == 1, "1")

        self.qp_model.addConstr(self.qp_w @ self.cov @ self.qp_w <= self.gamma*np.mean(self.cov) , "2")

        super().__init__()

    def _getModel(self):
        return self.qp_model, self.qp_w
    
    def setObj(self, r):
        self.c = r
        c_np   = self.c.detach().cpu().numpy()
        sense  = GRB.MAXIMIZE if self.maximize else GRB.MINIMIZE
        self.qp_model.setObjective(0, sense)
        self.qp_model.setObjective(c_np @ self.qp_w, sense)
    
    def solve(self):
        self.qp_model.optimize()
        if self.qp_model.status != GRB.OPTIMAL:
            print(self.qp_model.status)
            raise RuntimeError("Gurobi n’a pas trouvé d’optimum.")
        sol_np = self.qp_w.X.copy()
        sol_np[sol_np<1e-6] = 0      # nettoyage numérique
        obj_val = self.qp_model.ObjVal
        sol_t = torch.tensor(sol_np, dtype=self.c.dtype,
                                        device=self.c.device)
        return sol_t, obj_val 
