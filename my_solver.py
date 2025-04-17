import torch
import numpy as np

def lagrangian_decomposition(c, weights, capacity, mu):
    """
    Décomposition Lagrangienne pour le problème du sac à dos.
    c : coûts des objets
    weights : vecteur de poids des objets
    capacity : capacité du sac à dos
    mu : multiplicateur de Lagrange mu(c)

    On résoud Phi(X1) et on renvoie X1* sous forme de tenseur
    """

    
