import time
import random
import gurobipy as gp
import numpy as np
import torch
import pyepo
from pyepo.model.grb import knapsackModel
from opti_X_mu_CPU import OptimizationBatchModel
from knapsack.solver import solver_X_knapsack
from knapsack.data_import import ImportDataset
from models_class import CustomMLP

import os
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_dataset_file(fname, global_dim, num_feat, num_item, num_data, capacities, weights, obj, Z, c, x_star_array, keep=None, X=None, mu=None, seed=None):
    with open(fname, 'w') as f:
        # Constraints (unique for the whole dataset)
        if X is not None:
            f.write(f"{global_dim},{keep},{num_feat},{num_item},{num_data},{seed}\n")
        else :
            f.write(f"{global_dim},{num_feat},{num_item},{num_data},{seed}\n")
        for i in range(global_dim):
            line = str(int(capacities[i])) + "," + ",".join(str(int(w)) for w in weights[i][:-1]) + f",{int(weights[i][-1])}\n"
            f.write(line)
        # Data instances (features, cost, opt primal solution x, opt dual solution X, opt lagr. mult.)
        if X is not None:
            for i in range(num_data):
                line = str(obj[i]) + ","
                line += ",".join(str(z) for z in Z[i]) + ","
                line += ",".join(str(int(c)) for c in c[i]) + ","
                line += ",".join(str(int(x)) for x in x_star_array[i]) + ","
                line += ",".join(str(int(X)) for X in X[i]) + ","
                line += ",".join(str(mu) for mu in mu[i][:-1]) + f",{mu[i][-1]}\n"
                f.write(line)
        else:
            for i in range(num_data):
                line = str(obj[i]) + ","
                line += ",".join(str(z) for z in Z[i]) + ","
                line += ",".join(str(int(c)) for c in c[i]) + ","
                line += ",".join(str(int(x)) for x in x_star_array[i][:-1]) + f",{int(x_star_array[i][-1])}\n"
                f.write(line)

def gen_base_data(num_data_train, num_data_eval, num_data_test, num_feat, num_items, global_dim, 
                  deg=4, noise_width=0.5, verbose=False, seed=42):
    """
    Generate a base dataset for the knapsack problem.
    Shape (num_data, num_feat) for Z, (num_data, num_items) for c and x_star_array.
    The dataset is split into training, evaluation, and test sets.
    The primal solution x* is computed exactly for each instance.
    Args:
        num_data_train (int): Number of training data points.
        num_data_eval (int): Number of evaluation data points.
        num_data_test (int): Number of test data points.
        num_feat (int): Number of features.
        num_items (int): Number of items in the knapsack.
        global_dim (int): Number of constraints.
        deg (int): Degree of the polynomial for data generation.
        num_iter (int): Number of iterations for the optimization of mu.
        noise_width (float): Width of the noise for data generation.
        convergence (float): Convergence criterion for the optimization.
    """   
    total_data = num_data_train + num_data_test + num_data_eval
    if verbose:
        print(f"➡ Generation of {total_data} instances ({num_data_train} train, {num_data_eval} eval, {num_data_test} test)")
        print(f"➡ Dimensions : {global_dim} constraints, {num_items} items, {num_feat} features")

    # Random data generation
    weights, Z, c = pyepo.data.knapsack.genData(total_data, num_feat, num_items, global_dim, deg=deg, noise_width=noise_width, seed=seed)
    c = c.astype(int)
    weights = weights.astype(int)
    capacities = (0.5 * np.sum(weights, axis=1)).astype(int)

    # Exact primal problem solving (x*)
    if verbose:
        print(" Exact solving x*...")
    x_star_list = []
    obj_list = []
    time1 = time.time()
    model = knapsackModel(weights=weights, capacity=capacities)
    time2 = time.time()
    for i in range(total_data):
        model.setObj(c[i])
        x_star, obj = model.solve()
        obj_list.append(obj)
        x_star_list.append(x_star)
        if i == num_data_train - 1:
            time3 = time.time()
    
    time4 = time.time()

    x_star_array = np.array(x_star_list)
    obj_array = np.array(obj_list)
    
    if verbose:
        print(" Exact solving done.")
    
    NN = CustomMLP([num_feat, num_items], dropout=0.2).to(device)
    NN.eval()
    obj_hat_list = []
    c_hat = NN(torch.tensor(Z[:num_data_train], dtype=torch.float32, device=device)).detach().cpu().numpy()
    for i in range(num_data_train):
        model = knapsackModel(weights=weights, capacity=capacities)
        model.setObj(c_hat[i])
        _, obj_hat = model.solve()
        obj_hat_list.append(obj_hat)
        
    relative_regret = ((obj_array[:num_data_train]-np.array(obj_hat_list)) / np.array(obj_array[:num_data_train])).mean()
    if verbose:
        print(f"Relative regret on training set with random linear NN : {relative_regret:.4f}", flush=True)

    with open(f"knapsack/datasets/stats/data_base_stat_{num_data_train}_{total_data}_{num_items}_{global_dim}_{deg}_{noise_width}.txt", mode="a") as f:
        line = f"{seed};{time2-time1:.6f};{time3-time2:.6f};{time4-time2:.6f},{relative_regret:.4f}\n"
        f.write(line)
    
    Z_train, Z_eval, Z_test = Z[:num_data_train], Z[num_data_train:num_data_train+num_data_eval], Z[num_data_train+num_data_eval:]
    c_train, c_eval, c_test = c[:num_data_train], c[num_data_train:num_data_train+num_data_eval], c[num_data_train+num_data_eval:]
    x_star_train, x_star_eval, x_star_test = x_star_array[:num_data_train], x_star_array[num_data_train:num_data_train+num_data_eval], x_star_array[num_data_train+num_data_eval:]
    obj_train, obj_eval, obj_test = obj_array[:num_data_train], obj_array[num_data_train:num_data_train+num_data_eval], obj_array[num_data_train+num_data_eval:]
    
    write_dataset_file(f"knapsack/datasets/train_base_{num_items}_{global_dim}_{num_feat}_{num_data_train}_{deg}_{noise_width}_{seed}.txt",
                       global_dim, num_feat, num_items, num_data_train,
                       capacities, weights, obj_train, Z_train, c_train, x_star_train, seed=seed)

    write_dataset_file(f"knapsack/datasets/eval_{num_items}_{global_dim}_{num_feat}_{num_data_eval}_{deg}_{noise_width}_{seed}.txt",
                       global_dim, num_feat, num_items, num_data_eval,
                       capacities, weights, obj_eval, Z_eval, c_eval, x_star_eval, seed=seed)
    
    write_dataset_file(f"knapsack/datasets/test_{num_items}_{global_dim}_{num_feat}_{num_data_test}_{deg}_{noise_width}_{seed}.txt",
                       global_dim ,num_feat, num_items, num_data_test,
                       capacities, weights, obj_test, Z_test, c_test, x_star_test, seed=seed)
    return time3 - time2
        
def add_X_mu_single(num_data_train, num_feat, num_items, global_dim, keep=0, 
             num_iter=10000, convergence=1e-8, tl=0., budget=0,
             monitor=False, verbose=False, seed=42):
    """
    Add the X and mu variables to the dataset.
    Need the base dataset to be generated beforehand.
    Args:
        num_data_train (int): Number of training data points.
        num_feat (int): Number of features.
        num_items (int): Number of items in the knapsack.
        global_dim (int): Number of constraints.
        keep (int): Number of constraints to keep in the main subproblem.
        deg (int): Degree of the polynomial for data generation.
        num_iter (int): Number of iterations for the optimization of mu.
        noise_width (float): Width of the noise for data generation.
        convergence (float): Convergence criterion for the optimization.
        monitor (bool): Whether to monitor the optimization process.
        verbose (bool): Whether to print verbose output.
    """
    
    input_train_txt = f"knapsack/datasets/train_base_{num_items}_{global_dim}_{num_feat}_{num_data_train}_{deg}_{noise_width}_{seed}.txt"
    output_train_txt = f"knapsack/datasets/train_{num_items}_{global_dim}_{num_feat}_{num_data_train}_{deg}_{noise_width}_{budget}_{seed}.txt"
    if verbose:
        print(f"Reading existing file : {input_train_txt}")
    if not os.path.isfile(input_train_txt):
        raise FileNotFoundError(f"Can't find '{input_train_txt}'.")
    
    ds = ImportDataset(input_train_txt, model=None, z_stats=None, base=True)
    gd, nf, ni, nd = ds.get_sizes()
    if gd != global_dim or nf != num_feat or ni != num_items or nd != num_data_train:
        raise ValueError("The dataset dimensions do not match the expected values.")
    
    obj = ds.get_obj(tensor=False)  # numpy array (num_data_train)
    capacities = ds.get_capacities(tensor=False)  # numpy array (global_dim,)
    weights    = ds.get_weights(tensor=False)     # numpy array (global_dim, num_item)
    Z_train       = ds.Z           # (num_data_train, num_feat)
    c_train       = ds.c           # (num_data_train, num_item)
    x_star_train  = ds.x           # (num_data_train, num_item)

    solvers = [solver_X_knapsack(np.expand_dims(weights[keep],axis=0), np.expand_dims(capacities[keep],axis=0))]
    for i in range(global_dim):
        if i != keep:
            solvers.append(
                solver_X_knapsack(np.expand_dims(weights[i],axis=0), np.expand_dims(capacities[i],axis=0))
            )
    
    if verbose:
        print(f" Optimisation of mu...", flush=True)
        
    begin_time = time.time()  
    optimizer_mu = OptimizationBatchModel(solvers)
    num_iter_done = optimizer_mu.optim_mu(
        c_batch=c_train,
        verbose=verbose,
        max_iter=num_iter,
        convergence=convergence,
        timelimit=tl*budget
    )
    end_time = time.time()

    # 5. Récupérer X et μ calculés
    X_batch  = optimizer_mu.get_X(tensor=False)   # shape (num_data_train, global_dim, num_item)
    mu_batch = optimizer_mu.get_mu(tensor=False)  # shape (num_data_train, global_dim-new_keep, num_item)
    vals = optimizer_mu.get_value().cpu().numpy() # shape (num_data_train)
    
    
    if monitor:
        obj_array = ds.get_obj(tensor=False)  # (num_data_train)
        with open(f"knapsack/datasets/stats/gap_{num_data_train}_{num_items}_{global_dim}_{keep}.txt", mode="a") as f:
            line = f"{seed};{tl};{budget};{end_time-begin_time:.2f};{num_iter_done}"
            rapport = (vals - obj_array)/torch.tensor(obj_array)
            for i in range(rapport.shape[0]):
                line += f"{rapport[i]};"
            line += f"{rapport[-1]}\n"
            f.write(line)

    # Extraire la première composante X[:,0,:] et aplatir μ
    X_principal = X_batch[:, 0, :]                # (num_data_train, num_item)
    mu_flat     = np.reshape(mu_batch, (num_data_train, -1))  # (num_data_train, (global_dim-keep)*num_item)
    
    if verbose:
        print(f"Writing on file {output_train_txt}", flush=True)
    write_dataset_file(
        output_train_txt,
        global_dim=global_dim,
        keep=keep,
        num_feat=num_feat,
        num_item=num_items,
        num_data=num_data_train,
        capacities=capacities,
        weights=weights,
        Z=Z_train,
        obj=obj,
        c=c_train,
        x_star_array=x_star_train,
        X=X_principal,
        mu=mu_flat,
        seed=seed
    )
    if verbose:
        print("Dataset updated with X and mu variables.", flush=True)

def add_X_mu_multiple(num_data_train, num_feat, num_items, global_dim, 
             num_iter=10000, convergence=1e-8, timelimit=None,
             monitor=False, verbose=False, wandbarg=None):
    """
    Add the X and mu variables to the dataset.
    Need the base dataset to be generated beforehand.
    Args:
        num_data_train (int): Number of training data points.
        num_feat (int): Number of features.
        num_items (int): Number of items in the knapsack.
        global_dim (int): Number of constraints.
        keep (int): Number of constraints to keep in the main subproblem.
        deg (int): Degree of the polynomial for data generation.
        num_iter (int): Number of iterations for the optimization of mu.
        noise_width (float): Width of the noise for data generation.
        convergence (float): Convergence criterion for the optimization.
        monitor (bool): Whether to monitor the optimization process.
        verbose (bool): Whether to print verbose output.
    """
    if wandbarg is not None:
        import wandb
        #wandb.login(key="")  # Replace with your API key
        run = wandb.init(mode="offline", **wandbarg)
    
    input_train_txt = f"knapsack/datasets/train_base_{global_dim}_{num_feat}_{num_items}_{num_data_train}_{seed}.txt"
    output_train_txt = f"knapsack/datasets/train_{global_dim}_{keep}_{num_feat}_{num_items}_{num_data_train}_{seed}.txt"
    if verbose:
        print(f"Reading existing file : {input_train_txt}")
    if not os.path.isfile(input_train_txt):
        raise FileNotFoundError(f"Can't find '{input_train_txt}'.")
    
    ds = ImportDataset(input_train_txt, model=None, z_stats=None, base=True)
    gd, nf, ni, nd = ds.get_sizes()
    if gd != global_dim or nf != num_feat or ni != num_items or nd != num_data_train:
        raise ValueError("The dataset dimensions do not match the expected values.")
    
    obj = ds.get_obj(tensor=False)  # numpy array (num_data_train)
    capacities = ds.get_capacities(tensor=False)  # numpy array (global_dim,)
    weights    = ds.get_weights(tensor=False)     # numpy array (global_dim, num_item)
    Z_train       = ds.Z           # (num_data_train, num_feat)
    c_train       = ds.c           # (num_data_train, num_item)
    x_star_train  = ds.x           # (num_data_train, num_item)

    X_full = []
    mu_full = []
    vals_full = []

    if wandbarg is not None:
        import time
        begin_time = time.time()
        
    solvers = []
    for i in range(global_dim):
        solvers.append(
            solver_X_knapsack(np.expand_dims(weights[i],axis=0), np.expand_dims(capacities[i],axis=0))
        )
    optimizer_mu = OptimizationBatchModel(solvers)
    
    for i in range(global_dim):
        if verbose:
            print(f" Optimisation of mu keeping constraint {i}", flush=True)    
        optimizer_mu.optim_mu(
            c_batch=c_train,
            main_solver=i,  # On garde la contrainte i comme principale
            verbose=verbose,
            max_iter=num_iter,
            convergence=convergence,
            timelimit=timelimit//global_dim if timelimit is not None else None
        )
        
        # 5. Récupérer X et μ calculés
        X_batch  = optimizer_mu.get_X(tensor=False)   # shape (num_data_train, global_dim, num_item)
        X_full.append(X_batch[:, 0, :])  # On garde seulement la première composante X[:,0,:]
        mu_batch = optimizer_mu.get_mu(tensor=False)  # shape (num_data_train, global_dim-new_keep, num_item)
        mu_full.append(np.reshape(mu_batch, (num_data_train, -1)))  # On garde toutes les composantes de mu
        vals = optimizer_mu.get_value().cpu().numpy() # shape (num_data_train)
        vals_full.append(vals)  # On garde les valeurs de l'optimisation
        
                
    if wandbarg is not None:
        end_time = time.time()
        wandb.log({
            "time": end_time - begin_time,
            "num_iter": num_iter,
            "convergence": convergence,
            "num_data_train": num_data_train,
        })
        wandb.finish()
        
    if monitor:
        obj_array = ds.get_obj(tensor=False)  # (num_data_train)
        with open(f"knapsack/datasets/gap_{num_data_train}_{num_items}_{global_dim}_{num_iter}.txt", mode="a") as f:
            for i, vals in enumerate(vals_full):
                line = f"{i};"
                rapport = (vals - obj_array)/torch.tensor(obj_array)
                for i in range(rapport.shape[0]):
                    line += f"{rapport[i]};"
                line += f"{rapport[-1]}\n"
                f.write(line)

    X_full = np.hstack(X_full)               # (num_data_train, num_item)
    mu_full = np.hstack(mu_full)  # (num_data_train, (global_dim-keep)*num_item)
    
    if verbose:
        print(f"Writing on file {output_train_txt}", flush=True)
    write_dataset_file(
        output_train_txt,
        global_dim=global_dim,
        keep=-1,
        num_feat=num_feat,
        num_item=num_items,
        num_data=num_data_train,
        capacities=capacities,
        weights=weights,
        Z=Z_train,
        obj=obj,
        c=c_train,
        x_star_array=x_star_train,
        X=X_full,
        mu=mu_full
    )
    if verbose:
        print("Dataset updated with X and mu variables.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for generating a dataset with specified dimensions.")
    parser.add_argument('--keep', type=int, default=-1, help='Which constraint to keep in the main subproblem (-1 to generate multiple decomposition).')
    parser.add_argument('--budget', type=int, nargs="+", default=[1], help='Optimisation budget ( . * solving time)')
    parser.add_argument('--n_iter', type=int, default=100000, help='Number of iterations for the optimization of mu.')
    parser.add_argument('--conv', type=float, default=1e-4, help='Convergence stopping.')
    parser.add_argument('--monitor', type=bool, default=True, help='Whether to monitor the optimization process.')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print verbose output.')
    
    parser.add_argument('--n', type=int, nargs='+', default=[100], help='Number of items.')
    parser.add_argument('--dim', type=int, nargs='+', default=[5], help='Number of constraints.')
    parser.add_argument('--n_train', type=int, default=200, help='Number of training data points')
    parser.add_argument('--n_eval', type=int, default=100, help='Number of evaluation data points')
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test data points')
    parser.add_argument('--n_feat', type=int, default=12, help='Number of features')
    parser.add_argument('--noise', type=float, default=0.5, help='Convergence stopping.')
    parser.add_argument('--deg', type=int, default=4, help='')
    
    parser.add_argument('--seed', type=int, nargs="+", default=[42], help='Random seed for reproducibility.')

    # Parameters
    args = parser.parse_args()
    num_data_train = args.n_train
    num_data_eval = args.n_eval
    num_data_test = args.n_test
    num_feat = args.n_feat
    num_iter = args.n_iter
    num_item = args.n
    global_dim = args.dim
    budgets = args.budget
    convergence = args.conv
    keep = args.keep
    noise_width = args.noise
    deg = args.deg
    monitor = args.monitor
    verbose = args.verbose
    
    seeds = args.seed
 
    
    for n in num_item:
        for gd in global_dim:
            for seed in seeds:
                np.random.seed(seed)
                torch.manual_seed(seed)
                random.seed(seed)
                gp.setParam('Seed', seed)
                # Generate the base dataset
                if verbose:
                    print(f"Generating base dataset with {n} items, {gd} constraints, {deg} degree, noise width {noise_width}, {num_feat} features, {num_data_train} train, {num_data_eval} eval, {num_data_test} test, and {num_iter} iterations, seed {seed}.")
                tl = gen_base_data(num_data_train, num_data_eval, num_data_test, num_feat, n, gd, deg, noise_width, verbose, seed)
                if verbose:
                    print(f"solving time {tl}", flush=True)
                for budget in budgets:
                    # Add X and mu variables to the dataset
                    if verbose:
                        print(f"Adding X and mu variables to the dataset with time budget {tl*budget}.", flush=True)
                    if keep == -1:
                        add_X_mu_multiple(num_data_train, num_feat, n, gd, num_iter, convergence, tl, monitor, verbose)
                    else:
                        add_X_mu_single(num_data_train, num_feat, n, gd, keep, num_iter, convergence, tl, budget, monitor, verbose, seed=seed)