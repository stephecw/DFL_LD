import torch
import numpy as np
import gurobipy as gp
from pyepo.model.opt import optModel
from scipy.optimize import minimize

class Solveur_lin(optModel):
    def __init__(self, num_item):
        self.num_item = num_item
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
        sol = torch.zeros(self.num_item)
        sol[torch.argmax(self.c)] = 1
        return sol, None
    
class Solveur_quad(optModel):
    def __init__(self, num_item, cov, gamma):
        self.num_item = num_item
        self.cov = cov
        self.gamma = gamma
        super().__init__()

    def setObj(self, r):
        self.c = r.cpu().numpy()

    def _getModel(self):
        """
        Renvoie un *dummy* Gurobi model + variables binaires juste pour
        satisfaire l’interface requise par optModel.  
        Le solveur réel est dans self.solve().
        """
        # petit modèle bidon
        m = gp.Model()               # nécessite « import gurobipy as gp »
        x = [m.addVar(vtype=gp.GRB.BINARY, name=f"x{i}")
             for i in range(self.num_item)]
        m.update()
        return m, x                  # ⬅️ deux objets non‑None obligatoires

    def solve(self):
        def objective(x):
            return -np.dot(self.c, x)
        
        # Gradient de la fonction objective (Jacobien)
        def jacobian(x):
            return -self.c
        
        # Hessienne de la fonction objective (nulle pour une fonction linéaire)
        def hessian(x):
            return np.zeros((len(x), len(x)))
        
        # Jacobien de la contrainte
        def constraint_jacobian(x):
            return 2 * np.dot(self.cov, x)

        # Contrainte quadratique
        mean = np.mean(self.cov)
        def constraint(x):
            return self.gamma*mean - np.dot(x.T, np.dot(self.cov, x))

        # Contraintes de positivité
        bounds = [(0, None) for _ in range(self.num_item)]

        # Valeurs initiales
        x0 = np.ones(self.num_item)  # Utiliser des valeurs initiales raisonnables

        # Contrainte sous forme de dictionnaire
        constraints = ({'type': 'ineq', 'fun': constraint, 'jac':constraint_jacobian})

        # Résolution du problème d'optimisation
        res = minimize(objective, x0, jac=jacobian, hess=hessian, bounds=bounds, constraints=constraints, method='trust-constr', options={'maxiter': 100})

        return torch.Tensor(res.x), None
