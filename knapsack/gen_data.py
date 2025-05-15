import numpy as np
import torch
import pyepo
from pyepo.model.grb import knapsackModel
from opti_X_mu import OptimizationBatchModel


def f(x, c):
    return torch.dot(c, x)

def write_dataset_file(fname, dim, num_feat, num_item, num_data, capacities, weights, Z, c, x_star_array, X, mu):
    with open(fname, 'w') as f:
        # En-tête
        f.write(f"{dim},{num_feat},{num_item},{num_data}\n")
        for i in range(dim):
            line = str(int(capacities[i])) + "," + ",".join(str(int(w)) for w in weights[i][:-1]) + f",{int(weights[i][-1])}\n"
            f.write(line)
        for i in range(num_data):
            line = ""
            line += ",".join(str(Z[i][j]) for j in range(num_feat)) + ","
            line += ",".join(str(int(c[i][j])) for j in range(num_item)) + ","
            line += ",".join(str(int(x_star_array[i][j])) for j in range(num_item)) + ","
            line += ",".join(str(int(X[i][j])) for j in range(num_item)) + ","
            line += ",".join(str(mu[i][j]) for j in range(num_item*(dim-1) - 1)) + f",{mu[i][-1]}\n"
            f.write(line)


def gen_datafile(num_data_train, num_data_test, num_feat, num_item, dim, verbose=False):
    total_data = num_data_train + num_data_test

    if verbose:
        print(f"➡ Génération de {total_data} instances ({num_data_train} train, {num_data_test} test)")
        print(f"➡ Dimensions : {dim} contraintes, {num_item} items, {num_feat} features")

    # Données aléatoires (poids/capacités identiques pour tout le dataset)
    weights, Z, c = pyepo.data.knapsack.genData(total_data, num_feat, num_item, dim, deg=4, noise_width=0, seed=135)
    capacities = np.random.random() * 0.1 + 0.2 * np.sum(weights, axis=1)

    # Résolution exacte du problème (x*)
    if verbose:
        print(" Résolution exacte des x*...")
    x_star_list = []
    for i in range(total_data):
        model = knapsackModel(weights=weights, capacity=capacities)
        model.setObj(c[i])
        x_star, _ = model.solve()
        x_star_list.append(x_star)
    x_star_array = np.array(x_star_list)

    # Optimisation µ
    if verbose:
        print(" Optimisation de mu via GPU...")

    c_tensor = torch.tensor(c, dtype=torch.float32)
    optimizer = OptimizationBatchModel(
        num_item=num_item,
        dim=dim,
        c_batch=c_tensor,
        weights=weights,
        capacities=capacities,
        f=f
    )
    optimizer.optim_mu(verbose=verbose, max_iter=1000)

    X_tensor = optimizer.get_X()
    mu_tensor = optimizer.get_mu()

    X = X_tensor[:, 0, :].cpu().numpy().astype(int)
    mu = mu_tensor.view(total_data, -1).cpu().numpy()

    if verbose:
        print(f" Optimisation terminée (device: {torch.cuda.get_device_name()})")

    # Découpage
    Z_train, Z_test = Z[:num_data_train], Z[num_data_train:]
    c_train, c_test = c[:num_data_train], c[num_data_train:]
    x_star_train, x_star_test = x_star_array[:num_data_train], x_star_array[num_data_train:]
    X_train, X_test = X[:num_data_train], X[num_data_train:]
    mu_train, mu_test = mu[:num_data_train], mu[num_data_train:]

    # Sauvegarde
    write_dataset_file(f"datasets/train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt",
                       dim, num_feat, num_item, num_data_train,
                       capacities, weights, Z_train, c_train, x_star_train, X_train, mu_train)

    write_dataset_file(f"datasets/test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt",
                       dim, num_feat, num_item, num_data_test,
                       capacities, weights, Z_test, c_test, x_star_test, X_test, mu_test)

if __name__ == "__main__":
    
    import argparse

    # Définir les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Script de génération de dataset avec des dimensions spécifiées.")
    parser.add_argument('--n', type=int, nargs='+', default=[30], help='Nombre d\'item.')
    parser.add_argument('--dim', type=int, nargs='+', default=[5], help='Nombre de contraintes.')
    parser.add_argument('--n_train', type=int, default=500, help='Nombre de données d\'entrainement')
    parser.add_argument('--n_test', type=int, default=100, help='Nombre de données de test')
    parser.add_argument('--n_feat', type=int, default=200, help='Nombre de features')
    parser.add_argument('--n_iter', type=int, default=500, help='Nombre d\'itérations pour l\'optimisation de \mu. (0 pour ne pas l\'exécuter)')


    # Paramètres du dataset
    args = parser.parse_args()
    num_data_train = args.n_train
    num_data_test = args.n_test
    num_feat = args.n_feat
    num_iter = args.n_iter
    num_item = args.n
    dim = args.dim
    
    for n in num_item:
        for d in dim:
            gen_datafile(num_data_train, num_data_test, num_feat, n, d, verbose=True)