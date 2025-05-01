############# code adapté de : https://github.com/PredOpt/predopt-benchmarks/blob/main/Portfolio/Trainer/optimizer_module.py #############

import numpy as np
import cvxpy as cp
import cvxpylayers
from cvxpylayers.torch import CvxpyLayer

class cvxsolver:
    ''' 
    Implementation of QPTL with cvxpylayers and quadratic regularizer
    '''
    def __init__(self,cov, gamma, n_stocks = 50, alpha=1e-6,regularizer='quadratic'):
        '''
        regularizer: form of regularizer- either quadratic or entropic
        '''
        self.cov = cov
        self.gamma =  gamma
        self.n_stocks =  n_stocks
        self.alpha = alpha
        self.regularizer = regularizer
    



        x = cp.Variable(n_stocks)
        constraints = [x >= 0, cp.quad_form( x, cov ) <= gamma*np.mean(cov), cp.sum(x) <= 1]
        ### Original Model invoves inequality, We once tested  with Equality
        # constraints = [x >= 0, cp.quad_form( x, cov ) <= gamma, cp.sum(x) == 1]

        c = cp.Parameter(n_stocks)

        if self.regularizer=='quadratic':
            objective = cp.Minimize(-c @ x+ self.alpha*cp.pnorm(x, p=2))  
        elif self.regularizer=='entropic':
            objective = cp.Minimize(-c @ x -  self.alpha*cp.sum(cp.entr(x)) )
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[c], variables=[x])
    def solution(self, y):
              
        sol, = self.layer(y)
        return sol
    
### Normalement on peut pas le run. On ne peut pas gérer la contrainte quadrique

class cvxsolver_LD:
    ''' 
    Implementation of QPTL with cvxpylayers and quadratic regularizer
    '''
    def __init__(self,cov, gamma, n_stocks = 50, alpha=1e-6,regularizer='quadratic'):
        '''
        regularizer: form of regularizer- either quadratic or entropic
        '''
        self.cov = cov
        self.gamma =  gamma
        self.n_stocks =  n_stocks
        self.alpha = alpha
        self.regularizer = regularizer          # taille [n_stocks]
    



        x = cp.Variable(n_stocks)
        constraints = [x >= 0, cp.sum(x) <= 1]
        ### Original Model invoves inequality, We once tested  with Equality
        # constraints = [x >= 0, cp.sum(x) == 1]

        c = cp.Parameter(n_stocks)



        if self.regularizer=='quadratic':
            objective = cp.Minimize(-c @ x+ self.alpha*cp.pnorm(x, p=2))  
        elif self.regularizer=='entropic':
            objective = cp.Minimize(-c @ x -  self.alpha*cp.sum(cp.entr(x)) )
        problem = cp.Problem(objective, constraints)
        self.layer = CvxpyLayer(problem, parameters=[c], variables=[x])
    def solution(self, y):
              
        sol, = self.layer(y)
        return sol
