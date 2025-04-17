import torch
import numpy as np
from pyepo.model.grb import knapsackModel

def solve_main_problem(c, mu, weights, capacity):
    """
    On résoud Phi(X1) = max (c + mu_sum) · x | w·x ≤ cap de la décomposition Lagragienne
    Utilise pyEPO pour résoudre : max (c + mu_sum) · x | w·x ≤ cap_0

    c : Coûts des items
    mu : Multiplicateurs de Lagrange
    weights : Poids des items sur toute les dimensions
    capacity : Capacités des items sur toute les dimensions
    """
    weights_0 = weights[0]

    mu_sum = mu.sum(dim=1)-mu[0]

    capacity_0 = capacity[0]

    profit = c + mu_sum
    
    n_items = len(profit)
    
    # Création d’un modèle knapsack
    model = knapsackModel(n=n_items, budget=capacity_0, weight=weights_0)

    # Résolution
    x_opt = model.solve(profit.detach().cpu().numpy())  # pyEPO attend un numpy

    return torch.tensor(x_opt, dtype=torch.float32)

def solve_dual_subproblem(mu_i, weights_i, capacity_i):
    """
    On résout Psy(Xi) = max -μ_i·x | w_i·x ≤ cap_i de la décomposition Lagragienne
    Résout : max -μ_i·x | w_i·x ≤ cap_i avec pyEPO
    """
    profit = -mu_i
    n_items = len(profit)

    model = knapsackModel(n=n_items, budget=capacity_i, weight=weights_i)
    x_opt = model.solve(profit.detach().cpu().numpy())
    return torch.tensor(x_opt, dtype=torch.float32)





