from gen_data import gen_datafile, write_dataset_file
from data_import import ImportDataset
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pyepo.model.grb import knapsackModel
import gurobipy as gp

if __name__ == "__main__":
    # Paramètres du dataset
    # num_data_train = 500
    # num_data_test = 100
    # num_feat = 200
    # num_item = [30, 50, 100]
    # dim = [5,10]
    # param = [[10,100]]

    # for P in param:
    #     d = P[0]
    #     n = P[1]
    #     # Génération du dataset d'entraînement
    #     gen_datafile(num_data_train, num_data_test, num_feat, n, d, verbose=True)

    d_list = [5,10]

    n_list = [30,50,100]

    type_liste = ["train", "test"]

    num_data_train = 500
    num_data_test = 100

    
    for d in d_list:
        for n in n_list:
                for type in type_liste:
                    print (f"➡ Changement de {type} dataset ({d} contraintes, {n} items)")

                    if type == "train":
                        data = ImportDataset(f"datasets/train_{d}_200_{n}_{num_data_train}.txt")
                    else:
                        data = ImportDataset(f"datasets/test_{d}_200_{n}_{num_data_test}.txt")
                    Z = data.Z_tensor.cpu().numpy()
                    c = data.c_tensor.cpu().numpy()
                    x = data.x_tensor.cpu().numpy()
                    X = data.X_tensor.cpu().numpy()
                    mu = data.mu_tensor.cpu().numpy()

                    for i in range(Z.shape[0]):
                        model = knapsackModel(weights=[data.weights[0]], capacity=[data.capacities[0]])
                        model.setObj(c[i] + mu[i].sum(axis=0))
                        X1, _ = model.solve()
                        X[i] = X1
                    print("Optimisation terminée")
                    mu = mu.reshape(Z.shape[0], -1)
                    # Sauvegarde
                    if type == "train":
                        write_dataset_file(f"datasets/train_{d}_200_{n}_{num_data_train}.txt",
                                           data.dim, data.num_feat, data.num_item, data.num_data,
                                           data.capacities, data.weights, Z, c, x, X, mu)
                    else:
                        write_dataset_file(f"datasets/test_{d}_200_{n}_{num_data_test}.txt",
                                           data.dim, data.num_feat, data.num_item, data.num_data,
                                           data.capacities, data.weights, Z, c, x, X, mu)
                    
                    



