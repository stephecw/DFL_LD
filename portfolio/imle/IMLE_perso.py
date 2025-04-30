#!/usr/bin/env python
# coding: utf-8

from multiprocessing import process
from re import I
import numpy as np
import torch
from torch.autograd import Function

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.func.perturbed import implicitMLE, implicitMLEFunc
from pyepo.utlis import getArgs
from pyepo.func.utlis import sumGammaDistribution

class CustomIMLE(implicitMLE):
    """
    Une implémentation customisée de l'IMLE.
    Cette classe hérite de la classe implicitMLE de PyEPO.
    forward : retourne les solutions optimales de la borne LD : B(X, mu, c_hat), 
    avec c_hat et mu des batchs. Ces solutions sont calculées par Monte Carlo avec des perturbations.
    backward : calcule le gradient de LD par rapport à c_hat.
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, lambd=10,
                 distribution=sumGammaDistribution(kappa=5), two_sides=False,
                 processes=1, solve_ratio=1, dataset=None):
        """
        Args:
        optmodel (optModel): un modèle d'optimisation hérité de la classe optModel de PyEPO
        n_samples (int): le nombre d'échantillons de bruit pour Monte Carlo
        sigma (float): la temperature pour le bruit dans la distribution
        lambd (float): hyperparamètre pour l'interpolation lors de la différentiation
        distribution (sumGammaDistribution): la distribution de bruit à utiliser
        two_sides (bool): perturbation à deux côtés ou un seul
        processes (int): le nombre de processus pour la résolution
        """
        super().__init__(optmodel, n_samples, sigma, lambd, distribution, two_sides,
                         processes, solve_ratio, dataset)       
        self.imle = CustomIMLEFunc()

    def forward(self, pred_cost, mu):
        """
        Forward pass
        """
        # appel du forward de CustomIMLEFunc
        sols = self.imle.apply(pred_cost, mu, self) # Retourne les solutions optimales moyennées sur les perturbations
        return sols


class CustomIMLEFunc(Function):
    """
    Fonction de la classe CustomIMLE.
    Cette classe hérite de la classe Function de PyTorch.
    forward : retourne les solutions optimales de la borne LD : B(X, mu, c_hat),
    avec c_hat et mu des batchs. Ces solutions sont calculées par Monte Carlo avec des perturbations.
    backward : calcule le gradient de LD par rapport à c_hat.
    """

    @staticmethod
    def forward(ctx, pred_cost, mu, module):
        """
        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            mu (torch.tensor): a batch of Lagrangean multipliers
            module (optModule): implicitMLE module

        Returns:
            torch.tensor: predicted solutions
        """
        device = pred_cost.device
        cp = pred_cost.detach()
        
        # Perturbations
        noises = module.distribution.sample(size=(module.n_samples, *cp.shape))
        noises = torch.from_numpy(noises).to(device, dtype=torch.float32)
        ptb_c = cp + module.sigma * noises # Shape: (n_samples, batch_size, n_items) (broadcasting)
        
        # Trouve le X optimal de B(X, mu, ptb_c) pour chaque perturbation (n_samples*batch_size problèmes)
        ptb_sols = _solve(ptb_c, mu, module) # Shape: (batch_size, n_samples, n_items)
        
        # Monte Carlo : on fait la moyenne des solutions optimales sur les perturbations
        e_sol = ptb_sols.mean(dim=1) # Shape: (batch_size, n_items)
        
        # Save pour le backward
        ctx.save_for_backward(pred_cost, mu)
        ctx.noises = noises
        ctx.ptb_sols = ptb_sols
        ctx.module = module
        
        # Retourne les solutions optimales du batch
        return e_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for IMLE
        """
        # Récupération des tensors sauvegardés
        pred_cost, mu = ctx.saved_tensors
        noises = ctx.noises
        ptb_sols = ctx.ptb_sols
        module = ctx.module
        
        
        device = pred_cost.device
        cp = pred_cost.detach() # Shape: (batch_size, n_items)
        dl = grad_output.detach() # dB/dX1 (à propager) # Shape: (batch_size, n_items)
        
        # Cout perturbé d'un seul côté
        ptb_cp_pos = cp + module.lambd * dl + noises # Shape: (n_samples, batch_size, n_items) (broadcasting)
        
        # X optimal de B(X, mu, ptb_cp_pos)
        ptb_sols_pos = _solve(ptb_cp_pos, mu, module)
        
        # Si on utilise une perturbation à deux côtés
        if module.two_sides:
            # Cout perturbé d'un seul côté de l'autre coté
            ptb_cp_neg = cp - module.lambd * dl + noises
            ptb_sols_neg = _solve(ptb_cp_neg, mu, module)
            
            # two-side gradient
            grad = (ptb_sols_pos - ptb_sols_neg).mean(dim=1) / (2 * module.lambd)
        else:
            # single side gradient
            grad = (ptb_sols_pos - ptb_sols).mean(dim=1) / module.lambd
        return grad, None, None

def _solve(ptb_c, mu, module):
    """
    Retourne les solutions optimales X de B(X, mu, ptb_c)
    """
    device = ptb_c.device
    processes = module.processes
    optmodel = module.optmodel
    
    # taille du batch bruité
    n_samples, batch_size, n_item = ptb_c.shape
    
    # single-core
    if processes == 1:
        ptb_sols = torch.zeros((batch_size, n_samples, n_item), dtype=torch.float32, device=device)
        # Pour chaque element du batch
        for i in range(batch_size):
            # Pour chaque bruit sur l'élément
            for j in range(n_samples):
                # Résoudre le problème avec le solveur et la fonction de coût bruitée
                optmodel.setObj(ptb_c[j,i]+ mu[i].sum(axis=0)) # Coût bruité + somme des mu (borne LD)
                sol, _ = optmodel.solve() 
                ptb_sols[i,j] = torch.as_tensor(sol, dtype=torch.float32, device=device)
                
    # multi-core (PAS ENCORE ADAPTÉ POUR MU)
    else:
        # get class
        model_type = type(optmodel)
        # get args
        args = getArgs(optmodel)
        # parallel computing
        res = pool.amap(_solveWithObj4Par, ptb_c.permute(1, 0, 2),
                        [args] * ins_num, [model_type] * ins_num).get()
        # get solution
        ptb_sols = torch.stack(res, dim=0).to(device)
    return ptb_sols

def _solveWithObj4Par(perturbed_costs, args, model_type): #PAS ENCORE ADAPTÉ POUR MU
    """
    A global function to solve function in parallel processors

    Args:
        perturbed_costs (np.ndarray): costsof objective function with perturbation
        args (dict): optModel args
        model_type (ABCMeta): optModel class type

    Returns:
        list: optimal solution
    """
    # rebuild model
    optmodel = model_type(**args)
    # per sample
    sols = []
    for cost in perturbed_costs:
        # set obj
        optmodel.setObj(cost)
        # solve
        sol, _ = optmodel.solve()
        sols.append(sol)
    # to tensor
    sols = torch.tensor(sols, dtype=torch.float32)
    return sols
