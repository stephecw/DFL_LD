import numpy as np
import pyepo.data as data
from pyepo.model.grb import portfolioModel
from opti_X_mu import OptimizationModel
from joblib import Parallel, delayed


def write_dataset_file(fname, num_feat, num_item, num_data, cov, gamma, Z, r, x_star_array, X, mu):
    with open(fname, 'w') as f:
        # En-tête
        f.write(f"{num_data},{num_feat},{num_item},{gamma}\n")
        for i in range(num_item):
            f.write(",".join(str(cov[i][j]) for j in range(num_item)) + "\n")
        for i in range(num_data):
            line = ""
            line += ",".join(str(Z[i][j]) for j in range(num_feat)) + ","
            line += ",".join(str(r[i][j]) for j in range(num_item)) + ","
            line += ",".join(str(x_star_array[i][j]) for j in range(num_item)) + ","
            line += ",".join(str(X[i][j]) for j in range(num_item)) + ","
            line += ",".join(str(mu[i][j]) for j in range(num_item)) + f"\n"
            f.write(line)

def optimize_single_instance(r_i, cov, gamma, num_item, num_iter):
    optimizer = OptimizationModel(
        num_item=num_item,
        r=r_i,
        cov=cov,
        gamma=gamma,
    )
    optimizer.optim_mu(max_iter=num_iter)
    return optimizer.get_X0(), optimizer.get_mu()

def gen_datafile(num_data_train, num_data_test, num_feat, num_item, gam, num_iter, verbose=False):
    total_data = num_data_train + num_data_test

    if verbose:
        print(f"➡ Génération de {total_data} instances ({num_data_train} train, {num_data_test} test)")
        print(f"➡ Dimensions : {num_item} items, {num_feat} features")

    # Données aléatoires
    cov, Z, r = data.portfolio.genData(total_data, num_feat, num_item, deg=4, noise_level=1, seed=135)
    gamma = gam  # risk_level = gamma * mean(cov[i])
    
    # Résolution exacte du problème (x*)
    if verbose:
        print("Résolution exacte des x° ...", end=" ")
    x_star_list = []
    for i in range(total_data):
        model = portfolioModel(num_assets=num_item, covariance=cov, gamma=gamma) 
        model.setObj(r[i])
        x_star, _ = model.solve()
        x_star_list.append(x_star)
    x_star_array = np.array(x_star_list)
    if verbose:
        print("Done !")

    # Optimisation µ
    if verbose:
        print("Résolution approchée des mu et X° ...")

    # Parallélisation de l'optimisation de mu et X°
    results = Parallel(n_jobs=-1)(delayed(optimize_single_instance)(r[i], cov, gamma, num_item, num_iter) for i in range(total_data))

    X, mu = zip(*results)
    X = np.array(X)
    mu = np.array(mu)
    
    if verbose:
        print("Done !")

    # Découpage
    Z_train, Z_test = Z[:num_data_train], Z[num_data_train:]
    r_train, r_test = r[:num_data_train], r[num_data_train:]
    x_star_train, x_star_test = x_star_array[:num_data_train], x_star_array[num_data_train:]
    X_train, X_test = X[:num_data_train], X[num_data_train:]
    mu_train, mu_test = mu[:num_data_train], mu[num_data_train:]

    # Sauvegarde
    write_dataset_file(f"datasets/train_{num_item}_{num_data_train}_{num_feat}_{gamma}.txt",
                       num_feat, num_item, num_data_train,
                       cov, gamma, Z_train, r_train, x_star_train, X_train, mu_train)

    write_dataset_file(f"datasets/test_{num_item}_{num_data_train}_{num_feat}_{gamma}.txt",
                       num_feat, num_item, num_data_test,
                       cov, gamma, Z_test, r_test, x_star_test, X_test, mu_test)
    
    
    