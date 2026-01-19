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
    Solves   max/min   muᵀ·x
         s.t.          xᵀ·Cov·x ≤ γ·mean(Cov)
                       x ≥ 0                       (no Σx = 1 constraint here)

    Parameters
    ----------
    mu       : (n,)  profits (or costs if maximize=False)
    cov      : (n,n) symmetric positive semi-definite covariance matrix
    gamma    : float risk level
    maximize : bool  True  → maximize muᵀ·x
                       False → minimize muᵀ·x (useful for SPO+ on a minimization problem)
    tol      : float threshold for numerical cleanup

    Returns
    -------
    x_opt    : (n,) optimal solution as `np.ndarray`
    """
    n = mu.shape[0]

    # --- Build the Gurobi model ---
    m = gp.Model("quad_portfolio")
    m.setParam("OutputFlag", 0)        # silent

    x = m.addMVar(shape=n, lb=0.0, name="w")

    # quadratic risk constraint
    m.addConstr(x @ cov @ x <= gamma * cov.mean(), name="risk")

    # linear objective
    sense = GRB.MAXIMIZE if maximize else GRB.MINIMIZE
    m.setObjective(mu @ x, sense)

    # --- Solve ---
    m.optimize()

    #feas = m.computeIIS()
    if m.status == GRB.INFEASIBLE:
        print("The problem is infeasible.")

    if m.status != GRB.OPTIMAL:
        print(m.status)
        # raise RuntimeError("Gurobi did not find an optimal solution.")

    sol = x.X.copy()           # retrieve a np.ndarray
    sol[sol < tol] = 0.0       # optional numerical cleanup
    return sol


class Optimization_X_mu_portfolio:
    """Optimizes the LD bound for a combinatorial portfolio problem."""
    def __init__(self, num_item, c, cov, gamma, principal_lin = True):
        """
        Args:
        num_item (int): number of assets
        c_i (float array of shape (, num_item)): problem costs
        cov (float array of shape (num_item, num_item)): covariance matrix for the quadratic constraint
        gamma (float): risk_level in the quadratic constraint
        principal_lin (bool, optional): True to keep the linear constraint, False to keep the quadratic constraint. Defaults to True.
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
        """LD bound."""
        return np.dot(self.c, self.X[0]) + np.dot(self.mu, self.X[0] - self.X[1])
    
    def update_X(self):
        """Updates X°_1 and X°_2."""

        if self.lin:
            # Solve the linear-constraint sub-problem and place it based on the chosen main sub-problem
            self.X[0] = np.zeros(self.num_item, dtype=float)
            self.X[0][np.argmax(self.c + self.mu)] = 1.
            # Solve the quadratic-constraint sub-problem and place it based on the chosen main sub-problem
            self.X[int(self.lin)] = solve_sp(self.X[1], -self.mu, self.cov, self.gamma)
        else:
            # Solve the linear-constraint sub-problem and place it based on the chosen main sub-problem
            self.X[1] = np.zeros(self.num_item, dtype=float)
            self.X[1][np.argmax(-self.mu)] = 1
            # Solve the quadratic-constraint sub-problem and place it based on the chosen main sub-problem
            self.X[int(self.lin)] = solve_sp(self.X[1], self.c + self.mu, self.cov, self.gamma)

        
    def update_val(self):
        """Refreshes the value of the LD bound."""
        self.val_actuelle = self.B()
            
    def gradient(self):
        """Gradient of B w.r.t. mu. Requires X° that maximizes B for fixed mu."""
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
        Optimize mu with Adam.
        lr : float : learning rate
        beta1 : float : first moment parameter
        beta2 : float : second moment parameter
        eps : float : small value to avoid division by zero
        max_iter : int : maximum number of iterations
        """
        if mu0 is not None:
            self.mu = mu0
        self.adam_optimizer(self.gradient, lr, beta1, beta2, eps, max_iter, verbose)
    
    def get_mu(self):
        """
        Returns the current value of mu.
        """
        return self.mu
    
    def get_X0(self):
        """
        Returns the current value of X°_1.
        """
        self.update_X()
        return self.X[0]
    
    def get_X(self):
        """
        Returns the current value of X°.
        """
        self.update_X()
        return self.X
    
    def get_value(self):
        """
        Returns the current objective value.
        """
        self.update_val()
        return self.val_actuelle

    
    def info_sp_principal(self):
        """Prints which sub-problem is selected as the main one."""
        if self.lin:
            print("Main sub-problem: linear constraint")
        else:
            print("Main sub-problem: quadratic constraint")

    def optim_mu_batch(self, mu0_big=None, verbose=False, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
        """
        Optimize mu with Adam (batch mode).
        lr : float : learning rate
        beta1 : float : first moment parameter
        beta2 : float : second moment parameter
        eps : float : small value to avoid division by zero
        max_iter : int : maximum number of iterations
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
