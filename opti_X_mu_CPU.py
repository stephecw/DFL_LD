import numpy as np
import time
import torch
from joblib import Parallel, delayed


class OptimizationBatchModel:
    def __init__(self, solvers):
        """
        Batch-compatible optimization model for Lagrangian decomposition.

        Args:
            solvers: list of solvers to solve each sub-problem of LD
        """
        self.solvers = solvers
        self.num_pb = len(solvers)

    def update_val(self):
        Parallel(n_jobs=-1, backend="threading")(
            delayed(self.solve_X)(i) for i in range(self.num_pb)
        )
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
        Parallel(n_jobs=-1, backend="threading")(
            delayed(self.solve_X)(i) for i in range(self.num_pb)
        )
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
                    print(f"Time limit reached: {elapsed_time:.2f} seconds", flush=True)
                    break
            
            if verbose:
                print(f"    Iteration {t}/{max_iter} : {np.max(np.abs(lr * m_hat / (np.sqrt(v_hat) + eps)))}", flush=True)
            
            if np.max(np.abs(lr * m_hat / (np.sqrt(v_hat) + eps))) < convergence:
                print(f"Convergence reached", flush=True)
                break
        return t

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
        return self.adam_optimizer(verbose=verbose, **adam_args)

    def get_mu(self, tensor=True, device=torch.device("cpu")):
        return torch.tensor(self.mu, dtype=torch.float32, device=device) if tensor else self.mu

    def get_X0(self, tensor=True, device=torch.device("cpu")):
        self.X[0] = self.solvers[0](self.c + self.mu.sum(axis=1))
        return torch.tensor(self.X[:, 0], dtype=torch.int32, device=device) if tensor else self.X[:, 0]

    def get_X(self, tensor=True, device=torch.device("cpu")):
        Parallel(n_jobs=-1, backend="threading")(
            delayed(self.solve_X)(i) for i in range(self.num_pb)
        )
        return torch.tensor(self.X, dtype=torch.int32, device=device) if tensor else self.X

    def get_value(self, tensor=True, device=torch.device("cpu")):
        self.update_val()
        return torch.tensor(self.vals, dtype=torch.float32, device=device) if tensor else self.vals
    
class OptimizationBatchModel_serial:
    def __init__(self, solvers,method="adam"):
        """
        Batch-compatible optimization model for Lagrangian decomposition.

        Args:
            solvers: list of solvers to solve each sub-problem of LD
        """
        self.solvers = solvers
        self.num_pb = len(solvers)
        self.method = method

    def update_val(self):
        self.solve_X()
        self.vals = np.sum((self.c + self.mu.sum(axis=1)) * self.X[:,0], axis=1) - np.sum(self.mu*self.X[:, 1:], axis=(1,2))

    def solve_X(self):
        """
        Solve the X variables for all problems in the batch.
        This method updates self.X with the solutions for each sub-problem.
        X1 is solved using the main solver (self.idx_main),
        while Xi (i != idx_main) are solved using the corresponding solvers, preserving the order of solvers as specified in self.solvers.
        """
        self.X[:,0] = self.solvers[self.idx_main](self.c + self.mu.sum(axis=1))
        for idx_pb in range(self.idx_main):
            self.X[:,idx_pb+1] = self.solvers[idx_pb](-self.mu[:, idx_pb])
        for idx_pb in range(self.idx_main + 1, self.num_pb):
            self.X[:,idx_pb] = self.solvers[idx_pb](-self.mu[:, idx_pb-1])

    def gradient(self):
        """
        Calculate the gradient ∇B(μ) = X1 - Xi for the entire batch.
        """
        self.solve_X()
        return np.expand_dims(self.X[:, 0], axis=1) - self.X[:, 1:]

    def adam_optimizer(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=3000, convergence=1e-8, timelimit=None, verbose=False):
        """
        Adam batch for all problems simultaneously
        """
        if timelimit is not None:
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
                    print(f"Time limit reached: {elapsed_time:.4f} seconds", flush=True)
                    break
            
            if verbose:
                print(f"    Iteration {t}/{max_iter} : {np.max(np.abs(lr * m_hat / (np.sqrt(v_hat) + eps)))}", flush=True)
            
            if np.max(np.abs(lr * m_hat / (np.sqrt(v_hat) + eps))) < convergence:
                print(f"Convergence reached", flush=True)
                break
        return t
    
    def gradient_descent(self, lr=0.01, max_iter=3000, convergence=1e-8, timelimit=None, verbose=False):
        """
        Descente de gradient batch pour tous les problèmes simultanément
        """
        if timelimit is not None:
            start_time = time.time()

        for t in range(1, max_iter + 1):
            grad = self.gradient()
            update = lr * grad
            self.mu = self.mu - update

            if timelimit is not None:
                elapsed_time = time.time() - start_time
                if elapsed_time > timelimit:
                    print(f"Time limit reached: {elapsed_time:.4f} seconds", flush=True)
                    break

            if verbose:
                print(f"    Iteration {t}/{max_iter} : max update = {np.max(np.abs(update))}", flush=True)

            if np.max(np.abs(update)) < convergence:
                print("Convergence reached", flush=True)
                break

        return t

    def optim_mu(self, c_batch, main_solver=0, mu_init=None, verbose=False, **adam_args):
        self.idx_main = main_solver
        self.c = c_batch.clone().cpu().numpy() if isinstance(c_batch, torch.Tensor) else c_batch.copy()  # [B, n]
        batch_size, num_items = self.c.shape
        self.X = np.zeros((batch_size, self.num_pb, num_items), dtype=int)
        self.vals = np.zeros(batch_size, dtype=float)
        if mu_init is None:
            self.mu = np.ones((batch_size, self.num_pb - 1, num_items), dtype=float)
        else:
            self.mu = mu_init.clone().cpu().numpy() if isinstance(mu_init, torch.Tensor) else mu_init.copy()
        if self.method == "normal":
            return self.gradient_descent(lr=adam_args.get('lr', 0.01), max_iter=adam_args.get('max_iter', 3000), convergence=adam_args.get('convergence', 1e-8), timelimit=adam_args.get('timelimit'), verbose=verbose)
        elif self.method == "adam":
            return self.adam_optimizer(verbose=verbose, **adam_args)

    def get_mu(self, tensor=True, device=torch.device("cpu")):
        return torch.tensor(self.mu, dtype=torch.float32, device=device) if tensor else self.mu

    def get_X0(self, tensor=True, device=torch.device("cpu")):
        self.solve_X()
        return torch.tensor(self.X[:, 0], dtype=torch.int32, device=device) if tensor else self.X[:, 0]

    def get_X(self, tensor=True, device=torch.device("cpu")):
        self.solve_X()
        return torch.tensor(self.X, dtype=torch.int32, device=device) if tensor else self.X

    def get_value(self, tensor=True, device=torch.device("cpu")):
        self.update_val()
        return torch.tensor(self.vals, dtype=torch.float32, device=device) if tensor else self.vals
    