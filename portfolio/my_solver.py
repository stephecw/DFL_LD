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
        Juste pour satisfaire l’interface requise par optModel.  
        """
        m = gp.Model()               # nécessite « import gurobipy as gp »
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
            raise RuntimeError("Gurobi n’a pas trouvé d’optimum.")
        sol_np = self.qp_w.X.copy()
        obj_val = self.qp_model.ObjVal
        sol_t = torch.tensor(sol_np, dtype=self.c.dtype,
                                        device=self.c.device)
        return sol_t, obj_val 

from torch import Tensor

from joblib import Parallel, delayed


class BatchSolverLin:
    """Batch wrapper autour de Solveur_lin."""
    def __init__(self, num_item: int, maximize: bool = True, device: str = "cpu"):
        self.device = device
        # Un solveur (stateless) suffit : on réutilise la même instance
        self.solver = Solveur_lin(num_item, maximize=maximize)

    def __call__(self, c_batch: Tensor) -> Tensor:
        """
        c_batch: (B, n) torch.Tensor
        return: (B, n) torch.Tensor de 0/1
        """
        B, n = c_batch.shape
        # on va collecter chaque solution
        sols = []
        for i in range(B):
            c_i = c_batch[i]
            # met à jour l'objectif
            self.solver.setObj(c_i)
            sol_i, _ = self.solver.solve()      # sol_i: Tensor(n,)
            sols.append(sol_i)
        return np.stack(sols, axis = 0)      # (B, n)
    
from functools import partial

# On déclare un solveur global par processus
_GLOBAL_SOLVER = None

def init_solver(n_stocks, cov, gamma, maximize):
    """
    (Cette fonction n'est plus passée à Parallel,
    mais on la conserve pour ne pas changer le nom.)
    """
    global _GLOBAL_SOLVER
    _GLOBAL_SOLVER = Solveur_quad(n_stocks, cov, gamma, maximize=maximize)

def solve_one(c_i_np, n_stocks, cov, gamma):
    """
    Cette fonction est exécutée dans chaque process (ou dans le même process si n_jobs=1).
    Elle instancie le solveur une seule fois (lazy init) dans _GLOBAL_SOLVER
    puis réutilise ce même solveur pour chaque appel ultérieur dans ce process.
    """
    global _GLOBAL_SOLVER
    if _GLOBAL_SOLVER is None:
        _GLOBAL_SOLVER = Solveur_quad(n_stocks, cov, gamma)

    _GLOBAL_SOLVER.setObj(torch.from_numpy(c_i_np))
    sol, _ = _GLOBAL_SOLVER.solve()
    return sol


class BatchSolverQuad:
    """Wrapper qui lance chaque sous-problème quadratique dans un process séparé."""
    def __init__(self, n_stocks, cov, gamma, device="cpu", n_jobs=-1):
        # Réinitialisation du solveur global au moment de la création de l'instance
        global _GLOBAL_SOLVER
        _GLOBAL_SOLVER = None

        self.n_stocks = n_stocks
        # on stocke en numpy (pickle-friendly) si nécessaire
        self.cov = cov.detach().cpu().numpy() if torch.is_tensor(cov) else cov.copy()
        self.gamma = gamma
        self.device = device
        self.n_jobs = n_jobs

    def __call__(self, c_batch):
        """
        c_batch : (B, n) numpy.ndarray
        -> renvoie sol_np : (B, n) numpy.ndarray
        """
        # c_batch est déjà un numpy.ndarray, on l'utilise directement
        c_np = c_batch
        B = c_np.shape[0]

        # On crée une version « partielle » de solve_one,
        # qui lie n_stocks, cov, gamma, maximize
        bound_solve_one = partial(
            solve_one,
            n_stocks=self.n_stocks,
            cov=self.cov,
            gamma=self.gamma
        )

        # On lance Parallel sans initializer, en passant à bound_solve_one uniquement c_i_np
        sols = Parallel(
            n_jobs=self.n_jobs,
            backend="loky"
        )(
            delayed(bound_solve_one)(c_np[i]) for i in range(B)
        )

        # On reconstruit le tableau numpy des solutions
        sol_np = np.stack(sols, axis=0)  # forme (B, n)
        return sol_np

class BatchSolverExact:
    """
    Renvoie la solution exacte du sous problème avec contrainte quadratique grâce à la formule analytique
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
        -> renvoie sol_batch : (B, n) torch.Tensor
        """
        if not isinstance(c_batch, torch.Tensor):
            c_batch = torch.tensor(c_batch, dtype=torch.float32, device=self.device)
        else:
            c_batch = c_batch
        eps = 1e-8
        A = c_batch @ self.cov_inv 
        B = self.gamma * self.cov_mean / ((c_batch*A).sum(dim=1)+ eps)
        return torch.sqrt(B).unsqueeze(1) * A