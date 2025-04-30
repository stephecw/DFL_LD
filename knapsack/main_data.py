from gen_data import gen_datafile, write_dataset_file
from data_import import ImportDataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pyepo.model.grb import knapsackModel
import gurobipy as gp

if __name__ == "__main__":
    #Paramètres du dataset
    num_data_train = 500
    num_data_test = 100
    num_feat = 200
    num_item = [30, 50, 100]
    dim = [5,10]
    param = [[10,100]]

    for P in param:
        d = P[0]
        n = P[1]
        # Génération du dataset d'entraînement
        gen_datafile(num_data_train, num_data_test, num_feat, n, d, verbose=True)
                    
                    



