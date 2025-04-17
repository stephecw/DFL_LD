import numpy as np
import pyepo
import pyepo.data as data
import torch
from pyepo.model.grb import knapsackModel

def solve_kn1d(c, mu, weights, capacity, n_items, principal=False):
    """
    On résoud Phi(X1) = max (c + mu_sum) · x | w·x ≤ cap de la décomposition Lagragienne
    Utilise pyEPO pour résoudre : max (c + mu_sum) · x | w·x ≤ cap_0

    c : Coûts des items
    mu : Multiplicateurs de Lagrange
    weights : Poids des items
    capacity : Capacités des items
    n_items : Nombre d'items
    principal : bool : Si True, on ajoute les multiplicateurs de Lagrange au profit
    """
    profit = c + mu.sum(axis=0) if principal else -mu 
    
    # Création d’un modèle knapsack
    model = knapsackModel(n=n_items, budget=capacity, weight=weights)

    # Objectif
    model.setObj(profit)
    
    x_opt, val = model.solve()  # pyEPO attend un numpy

    return np.array(x_opt, dtype=int), val

class OptimizationModel:
    def __init__(self, num_item, dim, c, weight, capacity, f, mu0=None):
        """
        Modèle d'optimisation pour le problème du sac à dos multi-dimensionnel.
        c : np.array : Coûts des items
        num_item : int : Nombre d'items
        dim : int : Nombre de contraintes
        X0 : np.array : Valeur initiale de X°_1
        mu0 : np.array : Valeur initiale de mu
        """
        self.num_item = num_item
        self.dim = dim
        self.c = c
        self.weight = weight
        self.capacity = capacity
        self.f = f
        self.X = np.zeros((dim, num_item), dtype=int)
        self.mu = mu0 if mu0 is not None else np.ones((dim - 1, num_item), dtype=float)
        self.val_actuelle = 0
        
    def B(self):
        """Borne de la décomposition lagrangienne"""
        return self.f(self.X, self.c) + np.sum(self.mu*(self.X[0] - self.X[1:]))
            
    def gradient(self, mu_):
        """Gradient de B par rapport à mu. On a besoin de trouver X° qui maximise B à mu fixé"""
        self.X[0], self.val_actuelle = solve_kn1d(self.c, mu_, self.weight[0], self.capacity[0], self.num_item, principal=True)
        for i in range(1, self.dim, 1):
            X_i, val = solve_kn1d(self.c, mu_, self.weight[i], self.capacity[i], self.num_item)
            self.X[i] = X_i
            self.val_actuelle += val
        return self.X[0] - self.X[1:]

    def adam_optimizer(self, grad_func, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
        mu_ = self.mu
        m = np.zeros_like(mu_)
        v = np.zeros_like(mu_)

        for t in range(1, max_iter + 1):
            g = grad_func(self, mu_)

            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g ** 2)

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            mu_ -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if t % 100 == 0 or t == 1:
                print(f"Iter {t}, B(mu) = {self.val_actuelle:.6f}")

        self.mu = mu_
        
    def get_mu(self):
        return self.mu
    
    def optim_mu(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
        """
        Optimisation de mu par Adam.
        lr : float : Taux d'apprentissage
        beta1 : float : Premier paramètre de moment
        beta2 : float : Deuxième paramètre de moment
        eps : float : Petit nombre pour éviter la division par zéro
        max_iter : int : Nombre maximum d'itérations
        """
        self.adam_optimizer(self.gradient, lr, beta1, beta2, eps, max_iter)
    
    def get_X0(self):
        self.X[0], _ = solve_kn1d(self.c, self.mu, self.weight[0], self.capacity[0], self.num_item, principal=True)
        return self.X[0]
    
    def get_value(self, refresh=False):
        """
        Retourne la valeur actuelle de la fonction objectif.
        refresh : bool : Si True, on met à jour la valeur actuelle.
        """
        if refresh:
            self.val_actuelle = self.B()
        return self.val_actuelle