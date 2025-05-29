import numpy as np
import torch
import pyepo
from pyepo.model.grb import knapsackModel
from opti_X_mu import OptimizationBatchModel
from knapsack.solver import solver_X_1D_knapsack, solver_X_MD_knapsack

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_dataset_file(fname, global_dim, num_feat, num_item, num_data, capacities, weights, Z, c, x_star_array, X=None, mu=None):
    with open(fname, 'w') as f:
        # Constraints (unique for the whole dataset)
        f.write(f"{global_dim},{num_feat},{num_item},{num_data}\n")
        for i in range(global_dim):
            line = str(int(capacities[i])) + "," + ",".join(str(int(w)) for w in weights[i][:-1]) + f",{int(weights[i][-1])}\n"
            f.write(line)
        # Data instances (features, cost, opt primal solution x, opt dual solution X, opt lagr. mult.)
        if X is not None:
            for i in range(num_data):
                line = ",".join(str(Z[i][j]) for j in range(num_feat)) + ","
                line += ",".join(str(int(c[i][j])) for j in range(num_item)) + ","
                line += ",".join(str(int(x_star_array[i][j])) for j in range(num_item)) + ","
                line += ",".join(str(int(X[i][j])) for j in range(num_item)) + ","
                line += ",".join(str(mu[i][j].item()) for j in range(num_item*(global_dim-1) - 1)) + f",{mu[i][-1]}\n"
                f.write(line)
        else:
            for i in range(num_data):
                line = ",".join(str(Z[i][j]) for j in range(num_feat)) + ","
                line += ",".join(str(int(c[i][j])) for j in range(num_item)) + ","
                line += ",".join(str(int(x_star_array[i][j])) for j in range(num_item-1)) + f",{int(x_star_array[i][-1])}\n"
                f.write(line)

def gen_datafile(num_data_train, num_data_eval, num_data_test, num_feat, num_items, global_dim, keep ,num_iter, convergence, verbose=False):
    
    total_data = num_data_train + num_data_test + num_data_eval
    if verbose:
        print(f"➡ Generation of {total_data} instances ({num_data_train} train, {num_data_eval} eval, {num_data_test} test)")
        print(f"➡ Dimensions : {global_dim} constraints, {num_items} items, {num_feat} features")

    # Random data generation
    weights, Z, c = pyepo.data.knapsack.genData(total_data, num_feat, num_items, global_dim, deg=4, noise_width=0, seed=135)
    c = c.astype(int)
    weights = weights.astype(int)
    capacities = (np.random.random() * 0.1 + 0.2 * np.sum(weights, axis=1)).astype(int)

    # Exact primal problem solving (x*)
    if verbose:
        print(" Exact solving x*...")
    x_star_list = []
    for i in range(total_data):
        model = knapsackModel(weights=weights, capacity=capacities)
        model.setObj(c[i])
        x_star, _ = model.solve()
        x_star_list.append(x_star)
    x_star_array = np.array(x_star_list)

    # µ optimisation
    if verbose:
        print(" Optimisation of mu via GPU...")

    c = torch.tensor(c, dtype=torch.int32)
    Z_train, Z_eval, Z_test = Z[:num_data_train], Z[num_data_train:num_data_train+num_data_eval], Z[num_data_train+num_data_eval:]
    c_train, c_eval, c_test = c[:num_data_train], c[num_data_train:num_data_train+num_data_eval], c[num_data_train+num_data_eval:]
    x_star_train, x_star_eval, x_star_test = x_star_array[:num_data_train], x_star_array[num_data_train:num_data_train+num_data_eval], x_star_array[num_data_train+num_data_eval:]
    
    solvers = []
    if keep == 1:
        solvers= [solver_X_1D_knapsack(weights[0], capacities[0], device)]
    else:
        solvers = [solver_X_MD_knapsack(weights[:keep], capacities[:keep], device)]
    solvers += [solver_X_1D_knapsack(weights[i], capacities[i], device) for i in range(keep,global_dim)]
    optimizer_mu = OptimizationBatchModel(solvers, device)
    optimizer_mu.optim_mu(c_batch=c_train, verbose=verbose, max_iter=num_iter, convergence=convergence)
    X_train = optimizer_mu.get_X()
    mu_train = optimizer_mu.get_mu()

    X = X_train[:, 0, :].cpu()
    mu = mu_train.view(num_data_train, -1).cpu()

    if verbose:
        print(f" Optimisation done (device: {torch.cuda.get_device_name()})")
    # Save

    write_dataset_file(f"knapsack/datasets/train_{global_dim}_{keep}_{num_feat}_{num_items}_{num_data_train}.txt",
                       global_dim, num_feat, num_items, num_data_train,
                       capacities, weights, Z_train, c_train, x_star_train, X, mu)

    write_dataset_file(f"knapsack/datasets/eval_{global_dim}_{keep}_{num_feat}_{num_items}_{num_data_eval}.txt",
                       global_dim, num_feat, num_items, num_data_eval,
                       capacities, weights, Z_eval, c_eval, x_star_eval)
    
    write_dataset_file(f"knapsack/datasets/test_{global_dim}_{keep}_{num_feat}_{num_items}_{num_data_test}.txt",
                       global_dim, num_feat, num_items, num_data_test,
                       capacities, weights, Z_test, c_test, x_star_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for generating a dataset with specified dimensions.")
    parser.add_argument('--n', type=int, nargs='+', default=[30], help='Number of items.')
    parser.add_argument('--dim', type=int, nargs='+', default=[5], help='Number of constraints.')
    parser.add_argument('--n_train', type=int, default=500, help='Number of training data points')
    parser.add_argument('--n_eval', type=int, default=100, help='Number of evaluation data points')
    parser.add_argument('--n_test', type=int, default=200, help='Number of test data points')
    parser.add_argument('--n_feat', type=int, default=200, help='Number of features')
    parser.add_argument('--n_iter', type=int, default=500, help='Number of iterations for the optimization of mu. (0 to skip execution)')
    parser.add_argument('--keep', type=int, default=1, help='Number of constraints to keep in the main subproblem')
    parser.add_argument('--conv', type=float, default=1e-4, help='Convergence stopping.')

    # Parameters
    args = parser.parse_args()
    num_data_train = args.n_train
    num_data_eval = args.n_eval
    num_data_test = args.n_test
    num_feat = args.n_feat
    num_iter = args.n_iter
    num_item = args.n
    global_dim = args.dim
    convergence = args.conv
    
    for n in num_item:
        for gd in global_dim:
            gen_datafile(num_data_train, num_data_eval, num_data_test, num_feat, n, gd, num_iter, convergence, verbose=True)
