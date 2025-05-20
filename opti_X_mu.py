import torch

class OptimizationBatchModel:
    def __init__(self, solvers, device="cuda"):
        """
        Batch-compatible optimization model for Lagrangian decomposition.

        Args:
            solvers: list of solvers to solve each sub-problem of LD
        """
        self.solvers = solvers
        self.dim = len(solvers)
        self.device = device

    ### TO DO: Update val ###

    def solve_X(self):
        self.X[:,0] = self.solvers[0](self.c + self.mu.sum(dim=1))
        for i in range(1, len(self.solvers)):
            self.X[:,i] = self.solvers[i](-self.mu[:, i-1])

    def gradient(self):
        """
        Calculate the gradient ∇B(μ) = X1 - Xi for the entire batch.
        """
        self.solve_X()
        return self.X[:, 0].unsqueeze(1) - self.X[:, 1:]

    def adam_optimizer(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000, verbose=False, freq_verb=100):
        """
        Adam batch for all problems simultaneously
        """
        m = torch.zeros_like(self.mu)
        v = torch.zeros_like(self.mu)

        for t in range(1, max_iter + 1):
            grad = self.gradient()
            if verbose:
                print(f"    Iteration {t}/{max_iter} :")

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            self.mu = self.mu - lr * m_hat / (torch.sqrt(v_hat) + eps)

            if verbose and t % freq_verb == 0:
                print(f"→ Iter {t} | Mean B(mu): {self.vals.mean().item():.4f}")

    def optim_mu(self, c_batch, mu_init=None, verbose=False, **adam_args):
        self.c = c_batch.clone().to(self.device)  # [B, n]
        batch_size, num_items = self.c.shape
        self.X = torch.zeros((batch_size, self.dim, num_items), dtype=torch.int32, device=self.device)
        self.vals = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        if mu_init is None:
            self.mu = torch.ones((batch_size, self.dim - 1, num_items), dtype=torch.float32, device=self.device)
        else:
            self.mu = mu_init.to(self.device)
        self.adam_optimizer(verbose=verbose, **adam_args)

    def get_mu(self):
        return self.mu

    def get_X0(self):
        self.X[0] = self.solvers[0](self.c + self.mu.sum(dim=1))
        return self.X[0]

    def get_X(self):
        self.solve_X()
        return self.X

    def get_value(self):
        return self.vals

    def summary(self, top_k=5):
        print("\n====== Batch Optimization Summary ======")
        print(f"Batch size         : {self.batch_size}")
        print(f"Number of constraints: {self.dim}")
        print(f"Number of items     : {self.n_items}")
        print(f"Average bound B(μ)  : {self.vals.mean().item():.4f}")
        print(f"Top {top_k} B(μ)     : {self.vals[:top_k].cpu().numpy().round(2)}")
