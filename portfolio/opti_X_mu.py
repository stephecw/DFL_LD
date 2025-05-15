import numpy as np
import gurobipy as gp
from gurobipy import GRB

import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB
from joblib import Parallel, delayed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def solve_sp(X2, mu,cov,gamma, maximize= True, tol= 1e-6) :
    """
    Résout   max/min   muᵀ·x
           s.c.        xᵀ·Cov·x ≤ γ·mean(Cov)
                       x ≥ 0                         (pas de contrainte Σx = 1 ici)

    Paramètres
    ----------
    mu       : (n,)  profits (ou coûts si maximize=False)
    cov      : (n,n) matrice de covariance sym. définie‑positive
    gamma    : réel  niveau de risque
    maximize : bool  True  → maximise muᵀ·x
                       False → minimise muᵀ·x (utile pour SPO+ sur un problème de minimisation)
    tol      : float seuil de “nettoyage” numérique

    Retour
    ------
    x_opt    : (n,)  solution optimale comme `np.ndarray`
    """
    n = mu.shape[0]

    # --- Construction du modèle Gurobi ---
    m = gp.Model("quad_portfolio")
    m.setParam("OutputFlag", 0)        # silencieux

    x = m.addMVar(shape=n, lb=0.0, name="w")

    # contrainte de risque quadratique
    m.addConstr(x @ cov @ x <= gamma * cov.mean(), name="risk")

    # objectif linéaire
    sense = GRB.MAXIMIZE if maximize else GRB.MINIMIZE
    m.setObjective(mu @ x, sense)

    # --- Résolution ---
    m.optimize()

    #feas = m.computeIIS()
    if m.status == GRB.INFEASIBLE:
        print("Le problème est infaisable.")

    if m.status != GRB.OPTIMAL:
        print(m.status)
        #raise RuntimeError("Gurobi n’a pas trouvé d’optimum.")

    sol = x.X.copy()           # récupère un np.ndarray
    sol[sol < tol] = 0.0       # nettoyage numérique optionnel
    return sol


class Optimization_X_mu_portfolio:
    """Optimise la borne LD d'un problème de portfolio combinatoire"""
    def __init__(self, num_item, c, cov, gamma, principal_lin = True):
        """
        Args:
        num_item (int): nombre d'assets
        c_i (float array de taile ( ,num_item)): coûts du problème
        cov (float array de taille (num_item, num_item)): matrice de covariance de la contrainte quadratique
        gamma (float): risk_level dans la contrainte quadratique
        principal_lin (bool, optional): True pour conserver la contrainte linéaire, False pour conserver la contrainte quadratique. True par défaut
        """
        self.num_item = num_item
        self.c = c
        self.cov = cov
        self.gamma = gamma

        self.lin = principal_lin
        
        self.num_item = num_item
        self.X = np.zeros((2, num_item), dtype=float)
        self.mu = np.ones(num_item, dtype=float)

        self.val_actuelle = 0
        
    def B(self):
        """Borne de la LD"""
        return np.dot(self.c, self.X[0]) + np.dot(self.mu, self.X[0] - self.X[1])
    
    def update_X(self):
        """Met à jour X°_1 et X°_2"""

        if self.lin:
            # On résout le sous-problème avec contrainte linéaire, et on le place selon le choix de sous-problème principal
            self.X[0] = np.zeros(self.num_item, dtype=float)
            self.X[0][np.argmax(self.c + self.mu)] = 1.
            # On résout le sous-problème avec contrainte quadratique, et on le place selon le choix de sous-problème principal
            self.X[int(self.lin)] = solve_sp(self.X[1], -self.mu, self.cov, self.gamma)
        else:
            # On résout le sous-problème avec contrainte linéaire, et on le place selon le choix de sous-problème principal
            self.X[1] = np.zeros(self.num_item, dtype=float)
            self.X[1][np.argmax(-self.mu)] = 1
            # On résout le sous-problème avec contrainte quadratique, et on le place selon le choix de sous-problème principal
            self.X[int(self.lin)] = solve_sp(self.X[1], self.c + self.mu, self.cov, self.gamma)

        
    def update_val(self):
        """Actualise la valeur de la borne LD"""
        self.val_actuelle = self.B()
            
    def gradient(self):
        """Gradient de B par rapport à mu. On a besoin de trouver X° qui maximise B à mu fixé"""
        self.update_X()
        return self.X[0] - self.X[1]

    def adam_optimizer(self, grad_func, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000, verbose=False):
        m = np.zeros_like(self.mu)
        v = np.zeros_like(self.mu)
        for t in range(1, max_iter+1):
            if verbose:
                print(f"    Iteration {t}/{max_iter} :")
            g = grad_func()

            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g ** 2)

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            self.mu -= lr * m_hat / (np.sqrt(v_hat) + eps)

            if t % 500 == 0:
                self.update_val()
                print(f"        Iter {t}, B(mu) = {self.val_actuelle:.6f}")
    
    def optim_mu(self, mu0=None, verbose=False, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
        """
        Optimisation de mu par Adam.
        lr : float : Taux d'apprentissage
        beta1 : float : Premier paramètre de moment
        beta2 : float : Deuxième paramètre de moment
        eps : float : Petit nombre pour éviter la division par zéro
        max_iter : int : Nombre maximum d'itérations
        """
        if mu0 is not None:
            self.mu = mu0
        self.adam_optimizer(self.gradient, lr, beta1, beta2, eps, max_iter, verbose)
    
    def get_mu(self):
        """
        Retourne la valeur actuelle de mu.
        """
        return self.mu
    
    def get_X0(self):
        """
        Retourne la valeur actuelle de X°_1.
        """
        self.update_X()
        return self.X[0]
    
    def get_X(self):
        """
        Retourne la valeur actuelle de X°.
        """
        self.update_X()
        return self.X
    
    def get_value(self):
        """
        Retourne la valeur actuelle de la fonction objectif.
        """
        self.update_val()
        return self.val_actuelle

    
    def info_sp_principal(self):
        """Renvoie le type de sous-problème choisi en tant que principal"""
        if self.lin:
            print("Sous-problème principal : Contrainte linéaire")
        else:
            print("Sous-problème principal : Contrainte quadratique")

    def optim_mu_batch(self, mu0_big=None, verbose=False, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
        """
        Optimisation de mu par Adam.
        lr : float : Taux d'apprentissage
        beta1 : float : Premier paramètre de moment
        beta2 : float : Deuxième paramètre de moment
        eps : float : Petit nombre pour éviter la division par zéro
        max_iter : int : Nombre maximum d'itérations
        """
        mu_list = Parallel(n_jobs=-1)(delayed(utile)(self, mu0=mu0_big[i], verbose=verbose, lr=lr, beta1=beta1, beta2=beta2, eps=eps, max_iter=max_iter) for i in range(mu0_big.shape[0]))
        mu_np = np.stack(mu_list)
        mu_torch = torch.tensor(mu_np, device=device, dtype=torch.float32)
        return mu_torch
    
def utile(opti_X_mu, mu0=None, verbose=False, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
    if mu0 is not None:
        opti_X_mu.mu = mu0
    opti_X_mu.optim_mu(lr=lr, beta1=beta1, beta2=beta2, eps=eps, max_iter=max_iter, verbose=verbose)
    return opti_X_mu.get_mu()

