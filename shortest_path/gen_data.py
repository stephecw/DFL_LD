
import numpy as np
import pyepo.data
import torch
import pyepo
from opti_X_mu import OptimizationBatchModel
from shortest_path.solver import solver_partial_shortest_path, solver_partial_shortest_path_batch
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getArcs(grid):
        """
        Taken from https://github.com/khalil-research/PyEPO/blob/main/pkg/pyepo/model/grb/shortestpath.py

        Returns:
            list: arcs
        """
        arcs = []
        for i in range(grid[0]):
            # edges on rows
            for j in range(grid[1] - 1):
                v = i * grid[1] + j
                arcs.append((v, v + 1))
            # edges in columns
            if i == grid[0] - 1:
                continue
            for j in range(grid[1]):
                v = i * grid[1] + j
                arcs.append((v, v + grid[1]))
        return arcs

def write_dataset_file(fname, global_dim, num_feat, num_item, num_data, capacities, weights, Z, c, x_star_array, X=None, mu=None):
    with open(fname, 'w') as f:
        # Constraints (unique for the whole dataset)
        f.write(f"{global_dim},{keep},{num_feat},{num_item},{num_data}\n")
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
                line += ",".join(str(mu[i][j].item()) for j in range(num_item*(global_dim-keep) - 1)) + f",{mu[i][-1]}\n"
                f.write(line)
        else:
            for i in range(num_data):
                line = ",".join(str(Z[i][j]) for j in range(num_feat)) + ","
                line += ",".join(str(int(c[i][j])) for j in range(num_item)) + ","
                line += ",".join(str(int(x_star_array[i][j])) for j in range(num_item-1)) + f",{int(x_star_array[i][-1])}\n"
                f.write(line)

def gen_datafile(num_data_train, num_data_eval, num_data_test, num_feat, num_items, grid, partition, num_iter, convergence, verbose=False):
    total_data = num_data_train + num_data_test + num_data_eval
    if verbose:
        print(f"➡ Generation of {total_data} instances ({num_data_train} train, {num_data_eval} eval, {num_data_test} test)")
        print(f"➡ {grid[0]}x{grid[1]} grid, {num_feat} features, {keep} constraints kept in the main subproblem")

    # Random data generation
    Z, c = pyepo.data.shortestpath.genData(total_data, num_feat, grid, deg=8, noise_width=0.5, seed=135)
    num_nodes = grid[0]*grid[1]
    arcs = getArcs(grid)
    A = np.zeros((num_nodes, len(arcs)), dtype=int)
    for i in range(num_nodes):
        for j, arc in enumerate(arcs):
            if i == arc[0]:
                A[i][j] = -1
            elif i == arc[1]:
                A[i][j] = 1
    b = np.array([-1] + [0]*(num_nodes-2) + [1])

    # Exact primal problem solving (x*)
    if verbose:
        print(" Exact solving x*...")
    x_star_list = []
    obj_list = []
    for i in range(total_data):
        model = solver_partial_shortest_path(A, b)
        model.setObj(c[i])
        x_star, obj = model.solve()
        obj_list.append(obj)
        x_star_list.append(x_star)
    x_star_array = np.array(x_star_list)
    obj_array = np.array(obj_list)

    # µ optimisation
    if verbose:
        print(" Optimisation of mu via GPU...")
    c = torch.tensor(c, dtype=torch.int32)
    Z_train, Z_eval, Z_test = Z[:num_data_train], Z[num_data_train:num_data_train+num_data_eval], Z[num_data_train+num_data_eval:]
    c_train, c_eval, c_test = c[:num_data_train], c[num_data_train:num_data_train+num_data_eval], c[num_data_train+num_data_eval:]
    x_star_train, x_star_eval, x_star_test = x_star_array[:num_data_train], x_star_array[num_data_train:num_data_train+num_data_eval], x_star_array[num_data_train+num_data_eval:]
    
    solvers = []
    count = 0
    for n in partition:
        solvers += [solver_partial_shortest_path_batch(A[count:count+n], b[count:count+n], device)]
        count+=n
    optimizer_mu = OptimizationBatchModel(solvers, device)
    optimizer_mu.optim_mu(c_batch=c_train, verbose=verbose, max_iter=num_iter, convergence=convergence)
    X_train = optimizer_mu.get_X()
    mu_train = optimizer_mu.get_mu()
    vals = optimizer_mu.get_value().cpu().numpy()
    
    with open(f"knapsack/datasets/trainset_gap/gap_{num_items}_{partition}_{num_iter}.txt", mode="w") as f:
        line = f""
        rapport = (vals - obj_array[:num_data_train])/torch.tensor(obj_array[:num_data_train])
        for i in range(rapport.shape[0]):
            line += f"{rapport[i]};"
        line += f"{rapport[-1]}\n"
        f.write(line)

    X = X_train[:, 0, :].cpu()
    mu = mu_train.view(num_data_train, -1).cpu()

    if verbose:
        print(f" Optimisation done (device: {torch.cuda.get_device_name()})")
    # Save

    write_dataset_file(f"knapsack/datasets/train_{num_feat}_{grid[0]}_{grid[1]}_{num_data_train}.txt",
                       num_feat, grid, partition, num_data_train,
                       Z_train, c_train, x_star_train, X, mu)

    write_dataset_file(f"knapsack/datasets/eval_{num_feat}_{grid[0]}_{grid[1]}_{num_data_eval}.txt",
                       num_feat, grid, partition, num_data_eval,
                       Z_eval, c_eval, x_star_eval)
    
    write_dataset_file(f"knapsack/datasets/test_{num_feat}_{grid[0]}_{grid[1]}_{num_data_test}.txt",
                       global_dim, keep ,num_feat, num_items, num_data_test,
                       capacities, weights, Z_test, c_test, x_star_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for generating a dataset with specified dimensions.")
    parser.add_argument('--grid', type=int, nargs='+', default=[5, 5], help='Grid.')
    parser.add_argument('--n_train', type=int, default=500, help='Number of training data points')
    parser.add_argument('--n_eval', type=int, default=100, help='Number of evaluation data points')
    parser.add_argument('--n_test', type=int, default=400, help='Number of test data points')
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
    if len(args.n) != 2:
        print("Invalid grid shape")
        raise
    grid = tuple(args.n)
    convergence = args.conv
    keep = args.keep
    
    for n in num_item:
        for gd in global_dim:
            gen_datafile(num_data_train, num_data_eval, num_data_test, num_feat, n, gd, keep,num_iter, convergence, verbose=True)