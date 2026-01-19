import numpy as np
import torch
import pyepo.data as data
from pyepo.model.grb import portfolioModel
from portfolio.my_solver import BatchSolverLin, BatchSolverQuad, BatchSolverExact
from opti_X_mu_CPU import OptimizationBatchModel
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_dataset_file(fname, num_feat, num_item, num_data, deg, cov, gamma, Z, c, x_star_array, X, mu):
    """Writes the generated dataset to a file with the following format:
    num_data, num_feat, num_item, gamma
    [cov]
    [Z]_1, [c]_1, [x]_1, [X]_1, [mu]_1
    ...
    [Z]_{num_data}, [c]_{num_data}, [x]_{num_data}, [X]_{num_data}, [mu]_{num_data}

    Args:
        fname (str): file path
        num_feat (int): number of features for cost prediction
        num_item (int): number of assets
        num_data (int): dataset size
        cov (float array of shape (num_item, num_item)): covariance matrix of the quadratic constraint
        gamma (float): risk_level in the quadratic constraint
        Z (float array of shape (num_data, num_feat)): features
        c (float array of shape (num_data, num_item)): costs
        x_star_array (float array of shape (num_data, num_item)): optimal solutions of the original problems
        X (float array of shape (num_data, num_item)): optimal solutions of the main LD sub-problems
        mu (float array of shape (num_data, num_item)): optimal Lagrange multipliers for the LD bounds
    """
    with open(fname, 'w') as f:
        # Header
        f.write(f"{num_data},{num_feat},{num_item},{deg},{gamma}\n")
        for i in range(num_item):
            f.write(",".join(str(cov[i][j]) for j in range(num_item)) + "\n")
        for i in range(num_data):
            line = ""
            line += ",".join(str(Z[i][j]) for j in range(num_feat)) + ","
            line += ",".join(str(c[i][j]) for j in range(num_item)) + ","
            line += ",".join(str(x_star_array[i][j]) for j in range(num_item)) + ","
            line += ",".join(str(X[i][j]) for j in range(num_item)) + ","
            line += ",".join(str(mu[i][j]) for j in range(num_item)) + f"\n"
            f.write(line)

def gen_datafile(num_data_train, num_data_val, num_data_test, num_feat, num_item, deg, gam, num_iter, principal_lin = True, verbose=False):
    """Generates (train/val/test) datasets for combinatorial portfolio problems,
    including the optimal solution and the optimal LD bound.
    Saves these datasets into text files.

    Args:
        num_data_train (int): training dataset size
        num_data_test (int): test dataset size
        num_feat (int): number of features for cost prediction
        num_item (int): number of assets
        gam (float): risk_level in the quadratic constraint
        num_iter (int): number of iterations in the subgradient descent to optimize the LD bound
        principal_lin (bool, optional): True to keep the linear constraint, False to keep the quadratic constraint. Defaults to True.
        verbose (bool, optional): Whether to print progress. Defaults to False.
    """
    total_data = num_data_train + num_data_val + num_data_test 

    if verbose:
        print(f"➡ Generating {total_data} instances ({num_data_train} train, {num_data_test} test)")
        print(f"➡ Dimensions: {num_item} items, {num_feat} features")
        print(f"➡ Polynomial degree: {deg}")

    # random data generation
    cov, Z, c = data.portfolio.genData(total_data, num_feat, num_item, deg=deg, noise_level=1, seed=135)
    gamma = gam  # risk_level = gamma * mean(cov[i])
    cov2 = 1e5*cov  # covariance used for the quadratic constraint
    

    # Exact solve of the primal problem (x*)
    if verbose:
        print("Exact solving of x* ...", end=" ")
    x_star_list = []
    for i in range(total_data):
        model = portfolioModel(num_assets=num_item, covariance=cov, gamma=gamma) 
        model.setObj(c[i])
        x_star, _ = model.solve()
        x_star_list.append(x_star)
    x_star_array = np.array(x_star_list)
    if verbose:
        print("Done!")
    
    # Split
    Z_train, Z_val, Z_test = Z[:num_data_train], Z[num_data_train:num_data_train + num_data_val], Z[num_data_train + num_data_val:]
    c_train, c_val, c_test = c[:num_data_train], c[num_data_train:num_data_train + num_data_val], c[num_data_train + num_data_val:]
    x_star_train, x_star_val, x_star_test = x_star_array[:num_data_train], x_star_array[num_data_train:num_data_train + num_data_val], x_star_array[num_data_train + num_data_val:]

    X_val = np.zeros((num_data_val, num_item), dtype=float)
    mu_val = np.zeros((num_data_val, num_item), dtype=float)
    X_test = np.zeros((num_data_test, num_item), dtype=float)
    mu_test = np.zeros((num_data_test, num_item), dtype=float)

    gamma_str = str(gamma).replace('.', '-')

    write_dataset_file(f"portfolio/datasets/validation_{num_item}_{num_data_val}_{num_feat}_{deg}_{gamma_str}.txt",
                       num_feat, num_item, num_data_val, deg,
                       cov, gamma, Z_val, c_val, x_star_val, X_val, mu_val)
    
    write_dataset_file(f"portfolio/datasets/test_{num_item}_{num_data_test}_{num_feat}_{deg}_{gamma_str}.txt",
                       num_feat, num_item, num_data_test, deg,
                       cov, gamma, Z_test, c_test, x_star_test, X_test, mu_test)


    # Optimize μ
    if verbose:
        print("Approximate solving of μ and X° ...")
    # Optimize μ and X°
    c_tensor = torch.tensor(c, dtype=torch.float32)
    lin_solver = BatchSolverLin(num_item, device)
    quad_solver = BatchSolverQuad(num_item, cov2, gamma, device)
    exact_quad_solver = BatchSolverExact(num_item, cov2, gamma, device)
    if principal_lin:
        solvers = [lin_solver, quad_solver]
    else :
        solvers = [quad_solver, lin_solver]
    optimizer = OptimizationBatchModel(solvers)
    optimizer.optim_mu(c_batch=c_tensor[0:num_data_train],verbose=verbose, max_iter=num_iter)

    X_tensor = optimizer.get_X()
    mu_tensor = optimizer.get_mu()

    X = X_tensor[:, 0, :].cpu().numpy().astype(float)
    mu = mu_tensor.view(num_data_train, -1).cpu().numpy()

    if verbose:
        print(f" Optimisation done (device: {device})")

    X_train = X
    mu_train = mu

    
    # print(f"x : {x_star_train[0]}")
    # print(f"X : {X_train[0]}")
    # print(f"mu : {mu_train[0]}")
    # margin = np.array([gamma*np.mean(cov) - np.dot(X[i].T, np.dot(cov, X[i])) for i in range(total_data)])
    # print(margin)
    # print(margin.mean())
    # print(margin.std())
    # print()
    # print(gamma*np.mean(cov))

    # Save
    
    fold = "/lin" if principal_lin else "/quad"
    write_dataset_file(f"portfolio/datasets/train_{num_item}_{num_data_train}_{num_feat}_{deg}_{gamma_str}.txt",
                       num_feat, num_item, num_data_train, deg,
                       cov, gamma, Z_train, c_train, x_star_train, X_train, mu_train)

    
if __name__ == "__main__":

    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Dataset generation script with specified dimensions.")
    parser.add_argument('--n', type=int, default=50, help="Number of items.")
    parser.add_argument('--gamma', type=float, default=2.25, help="Gamma.")
    parser.add_argument('--n_train', type=int, default=100, help="Number of training instances.")
    parser.add_argument('--n_validation', type=int, default=25, help="Number of validation instances.")
    parser.add_argument('--n_test', type=int, default=10000, help="Number of test instances.")
    parser.add_argument('--n_feat', type=int, default=5, help="Number of features.")
    parser.add_argument('--lin', type=int, default=0, help="1 to keep the linear constraint as the main sub-problem, 0 to keep the quadratic constraint.")
    parser.add_argument('--n_iter', type=int, default=500, help="Number of iterations for μ optimization. (0 to skip)")
    parser.add_argument('--deg', type=int, default=8, help="Polynomial degree for data generation.")


    # Dataset parameters
    args = parser.parse_args()
    num_data_train = args.n_train
    num_data_test = args.n_test
    num_data_val = args.n_validation
    num_feat = args.n_feat
    num_iter = args.n_iter
    num_item = args.n
    gamma = args.gamma
    deg = args.deg
    principal_lin = False if args.lin == 0 else True
    gen_datafile(num_data_train, num_data_val,num_data_test, num_feat, num_item, deg, gamma, num_iter, principal_lin = principal_lin, verbose=True)
    
    
