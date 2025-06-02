import torch

class OptimizationBatchModel:
    def __init__(self, solvers, device="cuda"):
        """
        Batch-compatible Lagrangian multipliers optimizer.
        """
        self.solvers = solvers
        self.dim = len(solvers)
        self.device = device

        # buffers (will be allocated in the first optim_mu call)
        self.mu    = None
        self.X     = None
        self.vals  = None
        self.c     = None
        self.m     = None
        self.v     = None

    def solve_X(self):
        # assumes self.X, self.c and self.mu are already allocated
        # and have correct shapes
        # X[:,0]  <- solver_0( c + sum(mu) )
        self.X[:, 0] = self.solvers[0](
            self.c + self.mu.sum(dim=1)
        )
        # X[:,i]  <- solver_i( - mu[:,i-1] )
        for i in range(1, self.dim):
            self.X[:, i] = self.solvers[i](
                -self.mu[:, i-1]
            )

    def gradient(self):
        """
        ∇B(μ) = X1 - Xi
        """
        self.solve_X()
        return self.X[:, 0].unsqueeze(1) - self.X[:, 1:]

    def adam_optimizer(self, lr=0.01, beta1=0.9, beta2=0.999,
                       eps=1e-8, max_iter=10000, convergence=1e-4,
                       verbose=False):
        """
        In-place Adam on self.mu, reusing self.m and self.v buffers.
        """
        # on first call, allocate m & v
        if self.m is None or self.m.shape != self.mu.shape:
            self.m = torch.zeros_like(self.mu)
            self.v = torch.zeros_like(self.mu)

        for t in range(1, max_iter + 1):
            grad = self.gradient()  # [B, dim-1, n]

            # update moments in-place
            self.m.mul_(beta1).add_(grad, alpha=1-beta1)
            self.v.mul_(beta2).addcmul_(grad, grad, value=1-beta2)

            # bias-correct
            m_hat = self.m / (1 - beta1**t)
            v_hat = self.v / (1 - beta2**t)

            step = m_hat.div_(v_hat.sqrt().add_(eps)).mul_(lr)
            self.mu.add_(-step)   # mu = mu - step

            if verbose and t % 100 == 0:
                max_step = step.abs().max().item()
                print(f"  iter {t}/{max_iter}, max-step {max_step:.3e}")

            if step.abs().max() < convergence:
                if verbose:
                    print("Converged.")
                break

    def optim_mu(self, c_batch, mu_init=None, verbose=False, **adam_args):
        """
        Batch‐shape might change from one call to the next, so we
        allocate or resize our buffers only if needed.
        """
        # copy c into self.c
        c = c_batch.to(self.device)
        B, n = c.shape

        # re-alloc or reuse self.c
        self.c = c

        # allocate or reuse X: [B, dim, n]
        if self.X is None or self.X.shape != (B, self.dim, n):
            self.X = torch.empty((B, self.dim, n),
                                 dtype=torch.float32,
                                 device=self.device)

        # allocate or reuse vals: [B]
        if self.vals is None or self.vals.shape != (B,):
            self.vals = torch.empty((B,),
                                    dtype=torch.float32,
                                    device=self.device)

        # allocate or reuse mu: [B, dim-1, n]
        if mu_init is None:
            if self.mu is None or self.mu.shape != (B, self.dim-1, n):
                # first time or shape changed: fill with ones
                self.mu = torch.ones((B, self.dim-1, n),
                                     dtype=torch.float32,
                                     device=self.device)
            else:
                # same shape: just reset to ones in-place
                self.mu.fill_(1.0)
        else:
            # user‐provided init: copy into buffer
            mu0 = mu_init.to(self.device)
            if self.mu is None or self.mu.shape != mu0.shape:
                self.mu = mu0.clone()
            else:
                self.mu.copy_(mu0)

        # now run Adam to update self.mu in-place
        self.adam_optimizer(verbose=verbose, **adam_args)

    def get_mu(self):
        return self.mu

    def get_X(self):
        self.solve_X()
        return self.X

    def summary(self, top_k=5):
        print("=== BatchOptimize Summary ===")
        print(f"  batch-size: {self.X.shape[0]}")
        print(f"  constraints: {self.dim}")
        print(f"  items:       {self.X.shape[2]}")
        print(f"  avg B(mu):   {self.vals.mean().item():.4f}")
        print(f"  top-{top_k} B: {self.vals[:top_k].cpu().numpy().round(3)}")