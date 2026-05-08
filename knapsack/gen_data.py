
import numpy as np
import torch
import pyepo
from pyepo.model.grb import knapsackModel
from opti_X_mu_CPU import OptimizationBatchModel_serial
from knapsack.solver import solver_X_knapsack
from knapsack.data_import import ImportDataset
import os

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_dataset_file(fname, global_dim, num_feat, num_item, num_data, deg,capacities, weights, obj, Z, c, x_star_array, keep=None, X=None, mu=None, main=None):
    with open(fname, 'w') as f:
        # Constraints (unique for the whole dataset)
        if X is not None:
            f.write(f"{global_dim},{keep},{main},{num_feat},{num_item},{num_data},{deg}\n")
        else :
            f.write(f"{global_dim},{num_feat},{num_item},{num_data},{deg}\n")
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
                  deg=4, noise_width=0.5, verbose=False, wandbarg=None):
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
    if wandbarg is not None:
        import wandb
        #wandb.login(key="")  # Replace with your API key
        run = wandb.init(mode="offline", **wandbarg)
        
    total_data = num_data_train + num_data_test + num_data_eval
    if verbose:
        print(f"➡ Generation of {total_data} instances ({num_data_train} train, {num_data_eval} eval, {num_data_test} test)")
        print(f"➡ Dimensions : {global_dim} constraints, {num_items} items, {num_feat} features")

    # Random data generation
    weights, Z, c = pyepo.data.knapsack.genData(total_data, num_feat, num_items, global_dim, deg=deg, noise_width=noise_width, seed=42)
    c = c.astype(int)
    weights = weights.astype(int)
    capacities = (0.5 * np.sum(weights, axis=1)).astype(int)

    # Exact primal problem solving (x*)
    if verbose:
        print(" Exact solving x*...")
    x_star_list = []
    obj_list = []
    if wandbarg is not None:
        import time
        begin_time = time.time()
    for i in range(total_data):
        model = knapsackModel(weights=weights, capacity=capacities)
        model.setObj(c[i])
        x_star, obj = model.solve()
        obj_list.append(obj)
        x_star_list.append(x_star)
        if i == num_data_train - 1 and wandbarg is not None:
            end_time = time.time()
            wandb.log({
                "time": end_time - begin_time,
                "num_data_train": num_data_train,
                "num_items": num_items,
                "global_dim": global_dim,
            })
    x_star_array = np.array(x_star_list)
    obj_array = np.array(obj_list)
    
    if verbose:
        print(" Exact solving done.")

    Z_train, Z_eval, Z_test = Z[:num_data_train], Z[num_data_train:num_data_train+num_data_eval], Z[num_data_train+num_data_eval:]
    c_train, c_eval, c_test = c[:num_data_train], c[num_data_train:num_data_train+num_data_eval], c[num_data_train+num_data_eval:]
    x_star_train, x_star_eval, x_star_test = x_star_array[:num_data_train], x_star_array[num_data_train:num_data_train+num_data_eval], x_star_array[num_data_train+num_data_eval:]
    obj_train, obj_eval, obj_test = obj_array[:num_data_train], obj_array[num_data_train:num_data_train+num_data_eval], obj_array[num_data_train+num_data_eval:]
    
    write_dataset_file(f"knapsack/datasets/train_base_{global_dim}_{num_feat}_{num_items}_{num_data_train}_{deg}.txt",
                       global_dim, num_feat, num_items, num_data_train,deg,
                       capacities, weights, obj_train, Z_train, c_train, x_star_train)

    write_dataset_file(f"knapsack/datasets/eval_{global_dim}_{num_feat}_{num_items}_{num_data_eval}_{deg}.txt",
                       global_dim, num_feat, num_items, num_data_eval,deg,
                       capacities, weights, obj_eval, Z_eval, c_eval, x_star_eval)
    
    write_dataset_file(f"knapsack/datasets/test_{global_dim}_{num_feat}_{num_items}_{num_data_test}_{deg}.txt",
                       global_dim ,num_feat, num_items, num_data_test,deg,
                       capacities, weights, obj_test, Z_test, c_test, x_star_test)
    
    if wandbarg is not None:
        return end_time - begin_time
    else:
        return None

def add_X_mu_single(num_data_train, num_feat, num_items, global_dim, deg, keep=1, main=0,
             num_iter=10000, convergence=1e-8, timelimit= None,
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
        main (int): Index of the main subproblem (default is 0).
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
    
    input_train_txt = f"knapsack/datasets/train_base_{global_dim}_{num_feat}_{num_items}_{num_data_train}_{deg}.txt"
    output_train_txt = f"knapsack/datasets/train_{global_dim}_{keep}_{main}_{num_feat}_{num_items}_{num_data_train}_{deg}.txt"
    if verbose:
        print(f"Reading existing file : {input_train_txt}")
    if not os.path.isfile(input_train_txt):
        raise FileNotFoundError(f"Can't find '{input_train_txt}'.")
    
    ds = ImportDataset(input_train_txt, device)
    gd, nf, ni, nd = ds.get_sizes()
    if gd != global_dim or nf != num_feat or ni != num_items or nd != num_data_train:
        raise ValueError("The dataset dimensions do not match the expected values.")
    
    obj = ds.get_obj(tensor=False)  # numpy array (num_data_train)
    capacities = ds.get_capacities(tensor=False)  # numpy array (global_dim,)
    weights    = ds.get_weights(tensor=False)     # numpy array (global_dim, num_item)
    Z_train       = ds.Z           # (num_data_train, num_feat)
    c_train       = ds.c           # (num_data_train, num_item)
    x_star_train  = ds.x           # (num_data_train, num_item)

    solvers = [
        solver_X_knapsack(np.expand_dims(weights[i],axis=0),
                          np.expand_dims(capacities[i],axis=0))
        for i in range(global_dim)
    ]

    if verbose:
        print(f" Optimisation of mu...", flush=True)
        
    if wandbarg is not None:
        import time
        begin_time = time.time()

    optimizer_mu = OptimizationBatchModel_serial(solvers)
    c_tensor = torch.tensor(c_train, dtype=torch.int32)
    optimizer_mu.optim_mu(
        c_batch=c_tensor,
        main_solver=main,
        verbose=verbose,
        max_iter=num_iter,
        convergence=convergence,
        timelimit=timelimit
    )
    if wandbarg is not None:
        end_time = time.time()
        wandb.log({
            "time": end_time - begin_time,
            "num_iter": num_iter,
            "convergence": convergence,
            "num_data_train": num_data_train,
        })
        wandb.finish()

    # 5. Retrieve computed X and μ
    X_batch  = optimizer_mu.get_X(tensor=False)   # shape (num_data_train, global_dim, num_item)
    mu_batch = optimizer_mu.get_mu(tensor=False)  # shape (num_data_train, global_dim-new_keep, num_item)
    vals = optimizer_mu.get_value().cpu().numpy() # shape (num_data_train)

    if monitor:
        obj_array = ds.get_obj(tensor=False)  # (num_data_train)
        with open(f"knapsack/datasets/gap_{num_data_train}_{num_items}_{global_dim}_{keep}_{main}_{num_iter}_{deg}.txt", mode="w") as f:
            line = f""
            rapport = (vals - obj_array)/torch.tensor(obj_array)
            for i in range(rapport.shape[0]-1):
                line += f"{rapport[i]};"
            line += f"{rapport[-1]}\n"
            f.write(line)

    # Extract the first component X[:,0,:] and flatten μ
    X_principal = X_batch[:, 0, :]                # (num_data_train, num_item)
    mu_flat     = np.reshape(mu_batch, (num_data_train, -1))  # (num_data_train, (global_dim-keep)*num_item)
    
    if verbose:
        print(f"Writing on file {output_train_txt}", flush=True)
    write_dataset_file(
        output_train_txt,
        global_dim=global_dim,
        keep=keep,
        main=main,
        num_feat=num_feat,
        num_item=num_items,
        num_data=num_data_train,
        deg=deg,
        capacities=capacities,
        weights=weights,
        Z=Z_train,
        obj=obj,
        c=c_train,
        x_star_array=x_star_train,
        X=X_principal,
        mu=mu_flat
    )
    if verbose:
        print("Dataset updated with X and mu variables.", flush=True)

def add_X_mu_multiple(num_data_train, num_feat, num_items, global_dim, deg,
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
    keep = -1  # Indicate that we are generating multiple decompositions
    if wandbarg is not None:
        import wandb
        #wandb.login(key="")  # Replace with your API key
        run = wandb.init(mode="offline", **wandbarg)
    
    input_train_txt = f"knapsack/datasets/train_base_{global_dim}_{num_feat}_{num_items}_{num_data_train}_{deg}.txt"
    output_train_txt = f"knapsack/datasets/train_{global_dim}_{keep}_{num_feat}_{num_items}_{num_data_train}_{deg}.txt"
    if verbose:
        print(f"Reading existing file : {input_train_txt}")
    if not os.path.isfile(input_train_txt):
        raise FileNotFoundError(f"Can't find '{input_train_txt}'.")
    
    ds = ImportDataset(input_train_txt, device)
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
    optimizer_mu = OptimizationBatchModel_serial(solvers)
    
    for i in range(global_dim):
        if verbose:
            print(f" Optimisation of mu keeping constraint {i}", flush=True)    
        optimizer_mu.optim_mu(
            c_batch=torch.tensor(c_train, dtype=torch.int32),
            main_solver=i,  # Keep constraint i as the main one
            verbose=verbose,
            max_iter=num_iter,
            convergence=convergence,
            timelimit=timelimit//global_dim if timelimit is not None else None
        )
        
        # 5. Retrieve computed X and μ
        X_batch  = optimizer_mu.get_X(tensor=False)   # shape (num_data_train, global_dim, num_item)
        X_full.append(X_batch[:, 0, :])  # Keep only the first component X[:,0,:]
        mu_batch = optimizer_mu.get_mu(tensor=False)  # shape (num_data_train, global_dim-new_keep, num_item)
        mu_full.append(np.reshape(mu_batch, (num_data_train, -1)))  # Keep all μ components
        vals = optimizer_mu.get_value().cpu().numpy() # shape (num_data_train)
        vals_full.append(vals)  # Keep optimization values

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
        with open(f"knapsack/datasets/gap_{num_data_train}_{num_items}_{global_dim}_{num_iter}_{deg}.txt", mode="a") as f:
            for i, vals in enumerate(vals_full):
                line = f"{i};"
                rapport = (vals - obj_array)/torch.tensor(obj_array)
                for k in range(rapport.shape[0]-1):
                    line += f"{rapport[k]};"
                line += f"{rapport[-1]}\n"
                f.write(line)

    X_full = np.hstack(X_full)               # (num_data_train, global_dim * num_items)
    mu_full = np.hstack(mu_full)  #  (num_data_train, global_dim * (global_dim - 1) * num_items)
    
    if verbose:
        print(f"Writing on file {output_train_txt}", flush=True)
    write_dataset_file(
        output_train_txt,
        global_dim=global_dim,
        deg=deg,
        keep=-1,
        main=-1,
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
    parser.add_argument('--n_iter', type=int, default=10000, help='Number of iterations for the optimization of mu.')
    parser.add_argument('--conv', type=float, default=1e-4, help='Convergence stopping.')
    parser.add_argument('--monitor', type=bool, default=True, help='Whether to monitor the optimization process.')
    parser.add_argument('--verbose', type=bool, default=True, help='Whether to print verbose output.')
    
    parser.add_argument('--n', type=int, nargs='+', default=[100], help='Number of items.')
    parser.add_argument('--dim', type=int, nargs='+', default=[10], help='Number of constraints.')
    parser.add_argument('--n_train', type=int, default=200, help='Number of training data points')
    parser.add_argument('--n_eval', type=int, default=100, help='Number of evaluation data points')
    parser.add_argument('--n_test', type=int, default=1000, help='Number of test data points')
    parser.add_argument('--n_feat', type=int, default=12, help='Number of features')
    parser.add_argument('--noise', type=float, default=0.5, help='Convergence stopping.')
    parser.add_argument('--deg', type=int, default=4, help='')
    parser.add_argument('--main', type=int, default=0, help='Index of the main subproblem (default is 0).')

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
    keep = args.keep
    main = args.main
    noise_width = args.noise
    deg = args.deg
    monitor = args.monitor
    verbose = args.verbose
    
    for n in num_item:
        for gd in global_dim:
            # Generate the base dataset
            if verbose:
                print(f"Generating base dataset with {n} items, {gd} constraints, {deg} degree, noise width {noise_width}, {num_feat} features, {num_data_train} train, {num_data_eval} eval, {num_data_test} test, and {num_iter} iterations.")
            wandbarg = {
                        'dir': "./",
                        'name': f"base_data_{n}_{gd}_{deg}_{noise_width}_{num_feat}_{num_data_train}_{num_data_eval}_{num_data_test}_{num_iter}",
                        'group': f"knapsack",
                        'job_type': f"base_data",
                        'config': {
                            "convergence": convergence
                                    }
                        }
            tl = gen_base_data(num_data_train, num_data_eval, num_data_test, num_feat, n, gd, deg, noise_width, verbose, wandbarg)

            # Add X and mu variables to the dataset
            if verbose:
                print(f"Adding X and mu variables to the dataset with {n} items, {gd} constraints, {num_feat} features, {num_data_train} train, {num_iter} iterations, convergence {convergence}, keep {keep}.")
            keep_str = f"{keep}" if keep != -1 else "multiple"
            wandbarg = {
                        'dir': "./",
                        'name': f"opti_X_mu_{n}_{gd}_{deg}_{keep_str}_{main}_{num_feat}_{num_data_train}_{num_iter}",
                        'group': f"knapsack",
                        'job_type': f"opti_X_mu",
                        'config': {
                            "convergence": convergence,
                            "timelimit": tl,
                            "keep": keep,
                            "num_iter": num_iter,
                            "num_data_train": num_data_train,
                            "num_feat": num_feat,
                            "num_items": n,
                            "global_dim": gd,
                                    }
                        }
            if keep == -1:
                add_X_mu_multiple(num_data_train, num_feat, n, gd, deg, num_iter, convergence, tl, monitor, verbose, wandbarg)
            else:
                add_X_mu_single(num_data_train, num_feat, n, gd, deg, keep, main, num_iter, convergence, tl, monitor, verbose, wandbarg)
