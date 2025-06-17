import numpy as np
import torch
from joblib import Parallel, delayed

def knapsack_dynamic_programming(weights, values, capacity):
    n = len(values)
    # Initialiser une matrice (n+1) x (capacity+1) avec des zéros
    dp = [[0 for x in range(capacity + 1)] for y in range(n + 1)]

    # Construire la matrice dp de bas en haut
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]])
            else:
                dp[i][w] = dp[i - 1][w]

    # La valeur maximale qui peut être obtenue
    max_value = dp[n][capacity]

    # Retrouver les objets sélectionnés
    w = capacity
    selected_items = [0] * n  # Initialiser une liste de zéros
    for i in range(n, 0, -1):
        if max_value <= 0:
            break
        if max_value != dp[i - 1][w]:
            selected_items[i - 1] = 1  # Marquer l'objet comme sélectionné
            max_value -= values[i - 1]
            w -= weights[i - 1]

    return selected_items


class OptimizationBatchModel:
    def __init__(self, solvers, weights, capacity):
        """
        Batch-compatible optimization model for Lagrangian decomposition.

        Args:
            solvers: list of solvers to solve each sub-problem of LD
        """
        self.solvers = solvers
        self.num_pb = len(solvers)
        self.weights = weights
        self.capacity = capacity
        self.num_items = weights.shape[1]

    def update_val(self):
        c = np.vstack([self.c + self.mu.sum(axis=1)]+[-self.mu[:, i-1] for i in range(1, self.num_pb)])
        res = Parallel(n_jobs=-1, backend="loky")(
            delayed(knapsack_dynamic_programming)(self.weights[i//self.num_items], c[i], self.capacity[i//self.num_items]) for i in range(self.num_pb*self.num_items)
        )
        self.X = np.array(res).reshape(-1,self.num_pb, self.num_items)
        self.vals = np.sum((self.c + self.mu.sum(axis=1)) * self.X[:,0], axis=1) - np.sum(self.mu*self.X[:, 1:], axis=(1,2))

    def solve_X(self, idx_pb):
        if idx_pb == 0:
            self.X[:,0] = self.solvers[0](self.c + self.mu.sum(axis=1))
        else:
            self.X[:,idx_pb] = self.solvers[idx_pb](-self.mu[:, idx_pb-1])

    def gradient(self):
        """
        Calculate the gradient ∇B(μ) = X1 - Xi for the entire batch.
        """
        c = np.vstack([self.c + self.mu.sum(axis=1)]+[-self.mu[:, i-1] for i in range(1, self.num_pb)])
        res = Parallel(n_jobs=-1, backend="loky")(
            delayed(knapsack_dynamic_programming)(self.weights[i//self.num_items], c[i], self.capacity[i//self.num_items]) for i in range(c.shape[0])
        )
        self.X = np.array(res).reshape(-1,self.num_pb, self.num_items)
        return np.expand_dims(self.X[:, 0], axis=1) - self.X[:, 1:]

    def adam_optimizer(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=3000, convergence=1e-8, timelimit=None, verbose=False):
        """
        Adam batch for all problems simultaneously
        """
        if timelimit is not None:
            import time
            start_time = time.time()
        m = np.zeros_like(self.mu)
        v = np.zeros_like(self.mu)

        for t in range(1, max_iter + 1):
            grad = self.gradient()

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            self.mu = self.mu - lr * m_hat / (np.sqrt(v_hat) + eps)
            
            if timelimit is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time > timelimit:
                    print(f"Time limit reached: {elapsed_time:.2f} seconds, i = {t}", flush=True)
                    break
            
            if verbose:
                print(f"    Iteration {t}/{max_iter} : {np.max(np.abs(lr * m_hat / (np.sqrt(v_hat) + eps)))}", flush=True)
            
            if np.max(np.abs(lr * m_hat / (np.sqrt(v_hat) + eps))) < convergence:
                print(f"Convergence reached", flush=True)
                break

    def optim_mu(self, c_batch, main_solver=0, mu_init=None, verbose=False, **adam_args):
        self.solvers = [self.solvers[main_solver]] + self.solvers[:main_solver] + self.solvers[main_solver + 1:]
        self.c = c_batch.clone().cpu().numpy() if isinstance(c_batch, torch.Tensor) else c_batch.copy()  # [B, n]
        batch_size, num_items = self.c.shape
        self.X = np.zeros((batch_size, self.num_pb, num_items), dtype=int)
        self.vals = np.zeros(batch_size, dtype=float)
        if mu_init is None:
            self.mu = np.ones((batch_size, self.num_pb - 1, num_items), dtype=float)
        else:
            self.mu = mu_init.clone().cpu().numpy() if isinstance(mu_init, torch.Tensor) else mu_init.copy()
        self.adam_optimizer(verbose=verbose, **adam_args)

    def get_mu(self, tensor=True, device=torch.device("cpu")):
        return torch.tensor(self.mu, dtype=torch.float32, device=device) if tensor else self.mu

    def get_X0(self, tensor=True, device=torch.device("cpu")):
        self.X[0] = self.solvers[0](self.c + self.mu.sum(axis=1))
        return torch.tensor(self.X[:, 0], dtype=torch.int32, device=device) if tensor else self.X[:, 0]

    def get_X(self, tensor=True, device=torch.device("cpu")):
        c = np.vstack([self.c + self.mu.sum(axis=1)]+[-self.mu[:, i-1] for i in range(1, self.num_pb)])
        res = Parallel(n_jobs=-1, backend="loky")(
            delayed(knapsack_dynamic_programming)(self.weights[i//self.num_items], c[i], self.capacity[i//self.num_items]) for i in range(self.num_pb*self.num_items)
        )
        self.X = np.array(res).reshape(-1,self.num_pb, self.num_items)
        return torch.tensor(self.X, dtype=torch.int32, device=device) if tensor else self.X

    def get_value(self, tensor=True, device=torch.device("cpu")):
        self.update_val()
        return torch.tensor(self.vals, dtype=torch.float32, device=device) if tensor else self.vals
    


class OptimizationSingleModel:
    def __init__(self, solvers):
        """
        Batch-compatible optimization model for Lagrangian decomposition.

        Args:
            solvers: list of solvers to solve each sub-problem of LD
        """
        self.solvers = solvers
        self.num_pb = len(solvers)

    def update_val(self):
        self.solve_X()
        self.vals = np.sum((self.c + self.mu.sum(axis=0)) * self.X[0]) - np.sum(self.mu*self.X[1:])

    def solve_X(self):
        self.X[0] = self.solvers[0](self.c + self.mu.sum(axis=0))
        for i in range(1, self.num_pb):
            self.X[i] = self.solvers[i](-self.mu[i-1])

    def gradient(self):
        """
        Calculate the gradient ∇B(μ) = X1 - Xi for the entire batch.
        """
        self.solve_X()
        return np.expand_dims(self.X[0], axis=0) - self.X[1:]

    def adam_optimizer(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=3000, convergence=1e-8, timelimit=None, verbose=False):
        """
        Adam batch for all problems simultaneously
        """
        if timelimit is not None:
            import time
            start_time = time.time()
        m = np.zeros_like(self.mu)
        v = np.zeros_like(self.mu)

        for t in range(1, max_iter + 1):
            grad = self.gradient()

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            self.mu = self.mu - lr * m_hat / (np.sqrt(v_hat) + eps)
            
            if timelimit is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time > timelimit:
                    print(f"Time limit reached: {elapsed_time:.2f} seconds", flush=True)
                    break
            
            if verbose:
                print(f"    Iteration {t}/{max_iter} : {np.max(np.abs(lr * m_hat / (np.sqrt(v_hat) + eps)))}", flush=True)
            
            if np.max(np.abs(lr * m_hat / (np.sqrt(v_hat) + eps))) < convergence:
                print(f"Convergence reached", flush=True)
                break

    def optim_mu(self, c_batch, main_solver=0, mu_init=None, verbose=False, **adam_args):
        print(c_batch)
        self.solvers = [self.solvers[main_solver]] + self.solvers[:main_solver] + self.solvers[main_solver + 1:]
        self.c = c_batch.clone().cpu().numpy() if isinstance(c_batch, torch.Tensor) else c_batch.copy()  # [B, n]
        num_items = self.c.shape[0]
        self.X = np.zeros((self.num_pb, num_items), dtype=int)
        self.vals = 0.
        if mu_init is None:
            self.mu = np.ones((self.num_pb - 1, num_items), dtype=float)
        else:
            self.mu = mu_init.clone().cpu().numpy() if isinstance(mu_init, torch.Tensor) else mu_init.copy()
        self.adam_optimizer(verbose=verbose, **adam_args)

    def get_mu(self, tensor=True, device=torch.device("cpu")):
        return torch.tensor(self.mu, dtype=torch.float32, device=device) if tensor else self.mu

    def get_X0(self, tensor=True, device=torch.device("cpu")):
        self.X[0] = self.solvers[0](self.c + self.mu.sum(axis=1))
        return torch.tensor(self.X[:, 0], dtype=torch.int32, device=device) if tensor else self.X[:, 0]

    def get_X(self, tensor=True, device=torch.device("cpu")):
        self.solve_X()
        return torch.tensor(self.X, dtype=torch.int32, device=device) if tensor else self.X

    def get_value(self, tensor=True, device=torch.device("cpu")):
        self.update_val()
        return torch.tensor(self.vals, dtype=torch.float32, device=device) if tensor else self.vals