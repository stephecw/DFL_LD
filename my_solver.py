import torch
import numpy as np
from pyepo.model.grb import knapsackModel
import gurobipy as gp
from pyepo.model.opt import optModel

def solve_main_problem(c, mu, weights, capacity):
    """
    On résoud Phi(X1) = max (c + mu_sum) · x | w·x ≤ cap de la décomposition Lagragienne
    Utilise pyEPO pour résoudre : max (c + mu_sum) · x | w·x ≤ cap_0

    c : Coûts des items
    mu : Multiplicateurs de Lagrange
    weights : Poids des items sur toute les dimensions
    capacity : Capacités des items sur toute les dimensions
    """
    mu_sum = mu.sum(dim=0)
    profit = c + mu_sum
    n_items = len(profit)
    profit = list(profit)
    
    # Création d’un modèle knapsack
    model = knapsackModel(weights=np.expand_dims(weights[0], axis = 0), capacity =[capacity[0]])

    model.setObj(profit)  # pyEPO attend un numpy

    # Résolution
    x_opt, _ = model.solve()  # pyEPO attend un numpy

    return torch.tensor(x_opt, dtype=torch.float32)


def solve_dual_subproblem(mu_i, weights_i, capacity_i):
    """
    On résout Psy(Xi) = max -μ_i·x | w_i·x ≤ cap_i de la décomposition Lagragienne
    Résout : max -μ_i·x | w_i·x ≤ cap_i avec pyEPO
    """
    profit = -mu_i
    n_items = len(profit)

    model = knapsackModel(capacity=capacity_i, weight=weights_i)

    model.set_objective(profit.detach().cpu().numpy(), sense="max")  # pyEPO attend un numpy

    x_opt, _ = model.solve()
    return torch.tensor(x_opt, dtype=torch.float32)


class CustomOptModel(optModel):
    def __init__(self, weights, capacities, mu_batch):
        self.weights = weights
        self.capacities = capacities
        self.mu_batch = mu_batch
        self.batch_size = mu_batch.shape[0]
        self.n_items = mu_batch.shape[2]
        self.profit_batch = None

        super().__init__()

    def setObj(self, profit):
        self.profit = profit

    def _getModel(self):
        """
        Renvoie un *dummy* Gurobi model + variables binaires juste pour
        satisfaire l’interface requise par optModel.  
        Le solveur réel est dans self.solve().
        """
        # petit modèle bidon
        m = gp.Model()               # nécessite « import gurobipy as gp »
        x = [m.addVar(vtype=gp.GRB.BINARY, name=f"x{i}")
             for i in range(self.n_items)]
        m.update()
        return m, x                  # ⬅️ deux objets non‑None obligatoires

    def solve(self):
        sol = []
        print("profit_batch : ", self.profit_batch)
        for i in range(self.batch_size):
            c_i = self.profit_batch[i]
            mu_i = self.mu_batch[i]
            x_opt = solve_main_problem(profit, mu_i, self.weights, self.capacities)
            sol.append(x_opt)
        return torch.stack(sol)
