import torch
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from pyepo.model.opt import optModel
from pyepo.model.grb import optGrbModel

class Solveur_lin_torch(optModel):
    def __init__(self, num_item, maximize = True):
        self.num_item = num_item
        self.maximize = maximize
        super().__init__()

    def setObj(self, r):
        self.c = r

    def _getModel(self):
        """
        Only here to satisfy the interface required by optModel.
        """
        m = gp.Model()               # requires `import gurobipy as gp`
        x = [m.addVar(vtype=gp.GRB.BINARY, name=f"x{i}")
             for i in range(self.num_item)]
        m.update()
        return m, x  

    def solve(self):
        best_idx = torch.argmax(self.c) if self.maximize else torch.argmin(self.c)
        sol = torch.zeros(self.num_item, dtype=torch.float32, device=self.c.device)
        sol[best_idx] = 1
        return sol, torch.dot(self.c, sol)
    
class Solveur_lin(optModel):
    def __init__(self, num_item, maximize=True):
        self.num_item = num_item
        self.maximize = maximize
        super().__init__()

    def setObj(self, r):
        if isinstance(r, np.ndarray):
            self.c = r.copy()
        else:
            self.c = r.detach().cpu().numpy().copy()

    def _getModel(self):
        m = gp.Model()
        x_vars = [m.addVar(vtype=gp.GRB.BINARY, name=f"x{i}") 
                  for i in range(self.num_item)]
        m.update()
        return m, x_vars

    def solve(self):
        if self.maximize:
            best_idx = int(np.argmax(self.c))
        else:
            best_idx = int(np.argmin(self.c))
        sol = np.zeros(self.num_item, dtype=float)
        sol[best_idx] = 1.0
        obj_val = float(np.dot(self.c, sol))
        return sol, obj_val
    

class Solveur_quad(optGrbModel):
    def __init__(self, n_stocks, cov, gamma):
        self.n_stocks = n_stocks
        self.cov      = cov
        self.gamma    = gamma
        super().__init__()
    def _getModel(self):
        m = gp.Model("qp_portfolio")
        m.setParam("OutputFlag", 0)
        m.modelSense = GRB.MAXIMIZE
        # addVars returns a tupledict indexed by 0..n_stocks-1
        w = m.addVars(
            range(self.n_stocks),
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="w"
        )
        # build quadratic expression for w' * cov * w
        qexpr = gp.QuadExpr()
        for i in range(self.n_stocks):
            for j in range(self.n_stocks):
                coeff = self.cov[i, j]
                if coeff != 0:
                    qexpr.add(w[i] * coeff * w[j])
        m.addQConstr(
            qexpr <= self.gamma * np.mean(self.cov),
            name="risk"
        )
        return m, w
    
class gb_portfolio_solver(optModel):
    '''
    Gurobi solver takes the price as parameter, return the solution of the maximizimization problem
    '''
    def __init__(self,  n_stocks, cov, gamma, maximize=True):
        self.n_stocks = n_stocks
        self.cov = cov
        self.gamma =  gamma
        self.maximize = maximize

        self.qp_model = gp.Model("qp")
        self.qp_model.setParam('OutputFlag', 0)

        self.qp_w = self.qp_model.addMVar(shape= n_stocks, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

        self.qp_model.addConstr(self.qp_w.sum() == 1, "1")
        ### Original Model invoves inequality, We once tested  with Equality
        # model.addConstr(sum(x) == 1, "1")

        self.qp_model.addConstr(self.qp_w @ self.cov @ self.qp_w <= self.gamma*np.mean(self.cov) , "2")

        super().__init__()

    def _getModel(self):
        return self.qp_model, self.qp_w
    
    def setObj(self, r):
        self.c = r
        c_np   = self.c.detach().cpu().numpy()
        sense  = GRB.MAXIMIZE if self.maximize else GRB.MINIMIZE
        self.qp_model.setObjective(0, sense)
        self.qp_model.setObjective(c_np @ self.qp_w, sense)
    
    def solve(self):
        self.qp_model.optimize()
        if self.qp_model.status != GRB.OPTIMAL:
            print(self.qp_model.status)
            raise RuntimeError("Gurobi did not find an optimal solution.")
        sol_np = self.qp_w.X.copy()
        obj_val = self.qp_model.ObjVal
        sol_t = torch.tensor(sol_np, dtype=self.c.dtype,
                                        device=self.c.device)
        return sol_t, obj_val 

from torch import Tensor

from joblib import Parallel, delayed


class BatchSolverLin:
    """Batch wrapper around Solveur_lin."""
    def __init__(self, num_item: int, maximize: bool = True, device: str = "cpu"):
        self.device = device
        # A (stateless) solver is enough: reuse the same instance
        self.solver = Solveur_lin(num_item, maximize=maximize)

    def __call__(self, c_batch: Tensor) -> Tensor:
        """
        c_batch: (B, n) torch.Tensor
        return: (B, n) torch.Tensor de 0/1
        """
        B, n = c_batch.shape
        # collect each solution
        sols = []
        for i in range(B):
            c_i = c_batch[i]
            # update the objective
            self.solver.setObj(c_i)
            sol_i, _ = self.solver.solve()      # sol_i: Tensor(n,)
            sols.append(sol_i)
        return np.stack(sols, axis = 0)      # (B, n)
    
from functools import partial

# Declare one global solver per process
_GLOBAL_SOLVER = None

def init_solver(n_stocks, cov, gamma, maximize):
    """
    (This function is no longer passed to Parallel,
    but it is kept to avoid changing the name.)
    """
    global _GLOBAL_SOLVER
    _GLOBAL_SOLVER = Solveur_quad(n_stocks, cov, gamma, maximize=maximize)

def solve_one(c_i_np, n_stocks, cov, gamma):
    """
    This function runs in each process (or in the same process if n_jobs=1).
    It instantiates the solver only once (lazy init) in _GLOBAL_SOLVER,
    then reuses that solver for subsequent calls within the same process.
    """
    global _GLOBAL_SOLVER
    if _GLOBAL_SOLVER is None:
        _GLOBAL_SOLVER = Solveur_quad(n_stocks, cov, gamma)

    _GLOBAL_SOLVER.setObj(c_i_np)
    sol, _ = _GLOBAL_SOLVER.solve()
    return sol


class BatchSolverQuad:
    """Wrapper that runs each quadratic sub-problem in a separate process."""
    def __init__(self, n_stocks, cov, gamma, device="cpu", n_jobs=-1):
        # Reset the global solver at instance creation time
        global _GLOBAL_SOLVER
        _GLOBAL_SOLVER = None

        self.n_stocks = n_stocks
        # store as numpy (pickle-friendly) if needed
        self.cov = cov.detach().cpu().numpy() if torch.is_tensor(cov) else cov.copy()
        self.gamma = gamma
        self.device = device
        self.n_jobs = n_jobs

    def __call__(self, c_batch):
        """
        c_batch : (B, n) numpy.ndarray
        -> returns sol_np : (B, n) numpy.ndarray
        """
        # c_batch is already a numpy.ndarray, use it directly
        c_np = c_batch
        B = c_np.shape[0]

        # Create a "partial" version of solve_one binding n_stocks, cov, gamma
        bound_solve_one = partial(
            solve_one,
            n_stocks=self.n_stocks,
            cov=self.cov,
            gamma=self.gamma
        )

        # Run Parallel without initializer; pass only c_i_np to bound_solve_one
        sols = Parallel(
            n_jobs=self.n_jobs,
            backend="loky"
        )(
            delayed(bound_solve_one)(c_np[i]) for i in range(B)
        )

        # Rebuild the numpy array of solutions
        sol_np = np.stack(sols, axis=0)  # forme (B, n)
        return sol_np

class BatchSolverExact:
    """
    Returns the exact solution of the quadratic-constraint sub-problem using the analytic formula.
    """
    def __init__(self, num_item, cov, gamma, device="cpu", n_jobs=-1):
        self.num_item = num_item
        self.gamma = gamma
        self.device = device
        self.n_jobs = n_jobs

        cov_t = cov if torch.is_tensor(cov) else torch.tensor(cov, dtype=torch.float32)
        self.cov = cov_t.to(self.device)

        self.cov_inv = torch.linalg.inv(self.cov)
        self.cov_mean = self.cov.mean()
    
    def __call__(self, c_batch: torch.Tensor) -> torch.Tensor:
        """
        c_batch : (B, n) torch.Tensor
        -> returns sol_batch : (B, n) torch.Tensor
        """
        if not isinstance(c_batch, torch.Tensor):
            c_batch = torch.tensor(c_batch, dtype=torch.float32, device=self.device)
        else:
            c_batch = c_batch
        eps = 1e-8
        A = c_batch @ self.cov_inv 
        B = self.gamma * self.cov_mean / ((c_batch*A).sum(dim=1)+ eps)
        return torch.sqrt(B).unsqueeze(1) * A
