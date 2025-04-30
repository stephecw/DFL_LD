import torch
import numpy as np
from pyepo.model.grb import knapsackModel
import gurobipy as gp
from pyepo.model.opt import optModel
from scipy.optimize import minimize

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


class Solveur_lin(optModel):
    def __init__(self, num_item):
        self.num_item = num_item
        super().__init__()

    def setObj(self, r):
        self.c = r

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
        sol = torch.zeros(self.num_item)
        sol[torch.argmax(self.r)] = 1
        return sol
    
class Solveur_quad(optModel):
    def __init__(self, num_item, cov, gamma):
        self.num_item = num_item
        self.cov = cov
        self.gamma = gamma
        super().__init__()

    def setObj(self, r):
        self.c = r

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
        mean = torch.mean(self.cov)
        def objective(x):
            return np.dot(self.r.detach().cpu().numpy(), x)

        # Contrainte quadratique
        def constraint(x):
            return self.gamma*mean - np.dot(x.T, np.dot(self.cov.detach().cpu().numpy(), x))

        # Contraintes de positivité
        bounds = [(0, None) for _ in range(self.num_item)]

        # Contrainte quadratique sous forme de dictionnaire
        constraints = ({'type': 'ineq', 'fun': constraint})

        # Valeurs initiales
        x0 = np.ones(self.num_item)  # Utiliser des valeurs initiales raisonnables

        # Résolution du problème
        res = minimize(objective, x0, bounds=bounds, constraints=constraints, method='trust-constr')
    
        return torch.Tensor(res)
