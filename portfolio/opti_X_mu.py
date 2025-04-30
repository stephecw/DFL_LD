import numpy as np

from scipy.optimize import minimize

def solve_sp(X_2, mu, cov, gamma):
    # Fonction objective à minimiser (opposé de la fonction à maximiser)
    def objective(x):
        return np.dot(mu, x)

    # Contrainte quadratique
    def constraint(x):
        return gamma*np.mean(cov) - np.dot(x.T, np.dot(cov, x))

    # Contraintes de positivité
    bounds = [(0, None) for _ in range(X_2.shape[0])]

    # Contrainte quadratique sous forme de dictionnaire
    constraints = ({'type': 'ineq', 'fun': constraint})

    # Valeurs initiales
    x0 = np.ones(X_2.shape[0])  # Utiliser des valeurs initiales raisonnables

    # Résolution du problème
    res = minimize(objective, x0, bounds=bounds, constraints=constraints, method='trust-constr')
    
    return res.x

class OptimizationModel:
    def __init__(self, num_item, r, cov, gamma):
        """
        Modèle d'optimisation pour le problème du sac à dos multi-dimensionnel.
        c : np.array : Coûts des items
        num_item : int : Nombre d'items
        dim : int : Nombre de contraintes
        X0 : np.array : Valeur initiale de X°_1
        mu0 : np.array : Valeur initiale de mu
        """
        self.num_item = num_item
        self.r = r
        self.cov = cov
        self.gamma = gamma
        self.X = np.zeros((2, num_item), dtype=int)
        self.mu = np.ones(num_item, dtype=float)
        self.val_actuelle = 0
        
    def B(self):
        """Borne de la décomposition lagrangienne"""
        return np.dot(self.r, self.X[0]) + np.dot(self.mu, self.X[0] - self.X[1])
    
    def update_X(self):
        """Met à jour X°_1 et X°_2"""
        # On résout le sous-problème principal pour obtenir X°_1
        self.X[0] = np.zeros(self.num_item)
        self.X[0][np.argmax(self.r + self.mu)] = 1
        
        # On résout le deuxième sous-problème X°_2
        self.X[1] = solve_sp(self.X[1], self.mu, self.cov, self.gamma)
        
    def update_val(self):
        # Actualisation de la valeur actuelle
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
                print(f"        Iter {t}, B(mu) = {self.val_actuelle:.6f}")
    
    def optim_mu(self, mu0=None, verbose=False, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iter=1000):
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