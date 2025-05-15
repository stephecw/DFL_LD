import numpy as np
import pyepo
import torch
import pyepo.data as data
import torch
from pyepo.model.grb import knapsackModel
import gurobipy as gp
from solver_LD_GPU import solve_knapsack_gpu_batch

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
    profit = list(profit)

    
    # Création d’un modèle knapsack
    model = knapsackModel(weights, capacity)

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
            
    def gradient(self, mu_, verbose=False):
        """Gradient de B par rapport à mu. On a besoin de trouver X° qui maximise B à mu fixé"""

        problem = [self.dim, self.num_item]
        problem += list(self.c)
        problem += list(self.capacity)
        problem += list(self.weight.flatten())
        problems = [problem]

        u_flat = mu_.flatten()
        u_tensor = torch.tensor(u_flat, dtype=torch.float32).cuda()

        value_var_solution = solve_knapsack_gpu_batch(problems, u_tensor)

        X_tensor = value_var_solution.view(self.dim, self.num_item)
        X_numpy = X_tensor.cpu().numpy().astype(int)

        for i in range(self.dim):
            self.X[i] = X_numpy[i]

        val = self.f(self.X[0], self.c)
        for i in range(1, self.dim):
            val += np.sum(mu_[i - 1] * (self.X[0] - self.X[i]))
        self.val_actuelle = val

        if verbose:
            print(f"\n    ➤ B(mu) = {val:.4f}")
            for i in range(self.dim):
                print(f"    X_{i+1} =", self.X[i])

        return self.X[0] - self.X[1:]

    def adam_optimizer(self, grad_func, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000, verbose=False):
        mu_ = self.mu
        m = np.zeros_like(mu_)
        v = np.zeros_like(mu_)

        for t in range(1, max_iter+1):
            if verbose:
                print(f"    Iteration {t}/{max_iter} :")
            g = grad_func(mu_, verbose)

            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g ** 2)

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            mu_ -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if t % 500 == 0:
                print(f"        Iter {t}, B(mu) = {self.val_actuelle:.6f}")

        self.mu = mu_
        
    def get_mu(self):
        return self.mu
    
    def optim_mu(self, verbose=False, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
        """
        Optimisation de mu par Adam.
        lr : float : Taux d'apprentissage
        beta1 : float : Premier paramètre de moment
        beta2 : float : Deuxième paramètre de moment
        eps : float : Petit nombre pour éviter la division par zéro
        max_iter : int : Nombre maximum d'itérations
        """
        self.adam_optimizer(self.gradient, lr, beta1, beta2, eps, max_iter, verbose)
    
    def get_X0(self):
        self.X[0], _ = solve_kn1d(self.c, self.mu, np.expand_dims(self.weight[0], axis = 0), [self.capacity[0]], self.num_item, principal=True)
        return self.X[0]
    
    def get_X(self):
        self.X[0], _ = solve_kn1d(self.c, self.mu, np.expand_dims(self.weight[0], axis = 0), [self.capacity[0]], self.num_item, principal=True)
        for i in range(1, self.dim, 1):
            X_i, _ = solve_kn1d(self.c, self.mu[i-1], np.expand_dims(self.weight[i], axis = 0), [self.capacity[i]], self.num_item)
            self.X[i] = X_i
        return self.X
    
    def get_value(self, refresh=False):
        """
        Retourne la valeur actuelle de la fonction objectif.
        refresh : bool : Si True, on met à jour la valeur actuelle.
        """
        if refresh:
            self.val_actuelle = self.B()
        return self.val_actuelle

class OptimizationBatchModel:
    def __init__(self, num_item, dim, c_batch, weights, capacities, mu_init=None, device="cuda"):
        """
        Batch-compatible optimization model for Lagrangian decomposition of multi-dimensional knapsack.

        Args:
            num_item : int — nombre d’items
            dim : int — nombre de contraintes
            c_batch : [B, n] — batch de vecteurs de coûts
            weights : [m, n] — poids (identiques pour tout le batch)
            capacities : [m] — capacités (identiques pour tout le batch)
            f : fonction objectif sur X1 (ex: lambda x, c: c @ x)
            mu_init : [B, m-1, n] — initialisation facultative
        """
        self.n_items = num_item
        self.dim = dim
        self.batch_size = c_batch.shape[0]
        self.device = device

        # Data
        self.c = c_batch.to(device)  # [B, n]
        self.weights = weights.clone().to(device) # [m, n]
        self.capacity = capacities.clone().to(device)  # [m]

        # Variables
        if mu_init is None:
            self.mu = torch.ones((self.batch_size, dim - 1, num_item), dtype=torch.float32, device=device)
        else:
            self.mu = mu_init.to(device)

        self.X = torch.zeros((self.batch_size, dim, num_item), dtype=torch.int32, device=device)
        self.vals = torch.zeros(self.batch_size, dtype=torch.float32, device=device)

        # Format problèmes pour solve_knapsack_gpu_batch
        self.problems = self._build_problem_list()

    def _build_problem_list(self):
        """
        Prépare le batch de problèmes au format [dim, n, c..., cap..., w...]
        """
        problems = []
        for i in range(self.batch_size):
            problem = [self.dim, self.n_items]
            problem += list(self.c[i].cpu().numpy())
            problem += list(self.capacity.cpu().numpy())
            problem += list(self.weights.cpu().numpy().flatten())
            problems.append(problem)
        return problems

    def gradient(self, mu, verbose=False):
        """
        Calcule le gradient ∇B(μ) = X1 - Xi pour tout le batch.
        """
        # µ : [B, m-1, n]
        u_flat = mu.view(self.batch_size * (self.dim - 1) * self.n_items)
        x_all = solve_knapsack_gpu_batch(self.problems, u_flat)  # [B * m * n]
        x_all = x_all.view(self.batch_size, self.dim, self.n_items)

        self.X = x_all

        # Calcul de la valeur de la borne duale B(μ)
        with torch.no_grad():
            for i in range(self.batch_size):
                c_i = self.c[i]
                x1 = x_all[i, 0].float()
                dual_term = torch.sum(mu[i] * (x1 - x_all[i, 1:].float()))
                primal_val = torch.dot(x1, c_i)
                self.vals[i] = primal_val + dual_term

        # if verbose:
        #     for i in range(self.batch_size):
        #         print(f"[{i}] B(mu) = {self.vals[i].item():.4f}")
        #         for j in range(self.dim):
        #             print(f"  X_{j+1} = {self.X[i, j].tolist()}")

        # Retourne le gradient : [B, m-1, n]
        return self.X[:, 0].unsqueeze(1) - self.X[:, 1:]

    def adam_optimizer(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000, verbose=False):
        """
        Adam batch pour tous les problèmes simultanément
        """
        mu = self.mu
        m = torch.zeros_like(mu)
        v = torch.zeros_like(mu)

        for t in range(1, max_iter + 1):
            grad = self.gradient(mu, verbose)
            if verbose:
                print(f"    Iteration {t}/{max_iter} :")

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            mu = mu - lr * m_hat / (torch.sqrt(v_hat) + eps)

            if verbose and t % 200 == 0:
                print(f"→ Iter {t} | Mean B(mu): {self.vals.mean().item():.4f}")

        self.mu = mu

    def optim_mu(self, verbose=False, **adam_args):
        """Wrapper propre"""
        self.adam_optimizer(verbose=verbose, **adam_args)

    def get_mu(self):
        return self.mu
    
    def get_X0(self):
        self.X[0], _ = solve_kn1d(self.c, self.mu, np.expand_dims(self.weight[0], axis = 0), [self.capacity[0]], self.num_item, principal=True)
        return self.X[0]

    def get_X(self):
        self.X[0], _ = solve_kn1d(self.c, self.mu, np.expand_dims(self.weight[0], axis = 0), [self.capacity[0]], self.num_item, principal=True)
        for i in range(1, self.dim, 1):
            X_i, _ = solve_kn1d(self.c, self.mu[i-1], np.expand_dims(self.weight[i], axis = 0), [self.capacity[i]], self.num_item)
            self.X[i] = X_i
        return self.X

    def get_value(self):
        return self.vals

    def summary(self, top_k=5):
        print("\n====== Résumé Batch Optimization ======")
        print(f"Batch size         : {self.batch_size}")
        print(f"Nb contraintes     : {self.dim}")
        print(f"Nb items           : {self.n_items}")
        print(f"Borne moyenne B(μ) : {self.vals.mean().item():.4f}")
        print(f"Top {top_k} B(μ)   : {self.vals[:top_k].cpu().numpy().round(2)}")