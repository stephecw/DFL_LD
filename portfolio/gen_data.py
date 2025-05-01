import numpy as np
import pyepo.data as data
from pyepo.model.grb import portfolioModel
from opti_X_mu import Optimization_X_mu_portfolio
from joblib import Parallel, delayed


def write_dataset_file(fname, num_feat, num_item, num_data, cov, gamma, Z, c, x_star_array, X, mu):
    """Écrit le dataset généré dans un fichier dans le format suivant :
    num_data, num_feat, num_item, gamma
    [cov]
    [Z]_1, [c]_1, [x]_1, [X]_1, [mu]_1
    ...
    [Z]_{num_data}, [c]_{num_data}, [x]_{num_data}, [X]_{num_data}, [mu]_{num_data}

    Args:
        fname (str): emplacement du fichier
        num_feat (int): nombre de feature pour la prédiction du coût
        num_item (int): nombre d'assets
        num_data (int): taille du dataset
        cov (float array de taille (num_item, num_item)): matrice de covariance de la contrainte quadratique
        gamma (float): risk_level dans la contrainte quadratique
        Z (float array de taille (num_data, num_feat)): features
        c (float array de taille (num_data, num_item)): côuts
        x_star_array (float array de taille (num_data, num_item)): solutions optimales des problèmes
        X (float array de taille (num_data, num_item)): solutions optimales des sous-problèmes principaux des LD
        mu (float array de taille (num_data, num_item)): multiplicateurs de Lagr optimaux des bornes LD
    """
    with open(fname, 'w') as f:
        # En-tête
        f.write(f"{num_data},{num_feat},{num_item},{gamma}\n")
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

def optimize_single_instance(c_i, cov, gamma, num_item, num_iter, principal_lin):
    """ Optimise la borne LD pour un problème donné. Voué à être parallélisé.
    Args:
        c_i (float array de taile ( ,num_item)): coûts du problème
        cov (float array de taille (num_item, num_item)): matrice de covariance de la contrainte quadratique
        gamma (float): risk_level dans la contrainte quadratique
        num_item (int): nombre d'assets
        num_iter (int): nombre d'itérations dans la descente de sous-gradient
        principal_lin (bool, optional): True pour conserver la contrainte linéaire, False pour conserver la contrainte quadratique. True par défaut

    Returns:
        float array de taille ( ,num_item): solution optimale du sous-problème principal de la LD
        float array de taille ( ,num_item): multiplicateur de Lagr optimal
    """
    optimizer = Optimization_X_mu_portfolio(
        num_item=num_item,
        c=c_i,
        cov=cov,
        gamma=gamma,
        principal_lin = principal_lin
    )
    optimizer.optim_mu(max_iter=num_iter)
    return optimizer.get_X0(), optimizer.get_mu()

def gen_datafile(num_data_train, num_data_test, num_feat, num_item, gam, num_iter, principal_lin = True, verbose=False):
    """Génère un dataset (train et test) de problèmes de portfolio combinatoires, 
    comprenant la solution optimale et la borne LD optimale.
    Enregistre ces datasets dans des fichiers textes.

    Args:
        num_data_train (int): taille du dataset d'entraînement
        num_data_test (int): taille du dataset de test
        num_feat (int): nombre de feature pour la prédiction du coût
        num_item (int): nombre d'assets
        gam (float): risk_level dans la contrainte quadratique
        num_iter (int): nombre d'itérations dans la descente de sous-gradient pour optimiser la borne LD
        principal_lin (bool, optional): True pour conserver la contrainte linéaire, False pour conserver la contrainte quadratique. True par défaut
        verbose (bool, optional): Affiche l'avancement de la génération. Defaults to False.
    """
    total_data = num_data_train + num_data_test

    if verbose:
        print(f"➡ Génération de {total_data} instances ({num_data_train} train, {num_data_test} test)")
        print(f"➡ Dimensions : {num_item} items, {num_feat} features")

    # Données aléatoires
    cov, Z, c = data.portfolio.genData(total_data, num_feat, num_item, deg=4, noise_level=1, seed=135)
    gamma = gam  # risk_level = gamma * mean(cov[i])
    
    # Résolution exacte du problème (x*)
    if verbose:
        print("Résolution exacte des x° ...", end=" ")
    x_star_list = []
    for i in range(total_data):
        model = portfolioModel(num_assets=num_item, covariance=cov, gamma=gamma) 
        model.setObj(c[i])
        x_star, _ = model.solve()
        x_star_list.append(x_star)
    x_star_array = np.array(x_star_list)
    if verbose:
        print("Done !")

    # Optimisation µ
    if verbose:
        print("Résolution approchée des mu et X° ...")
    # Parallélisation de l'optimisation de mu et X°
    results = Parallel(n_jobs=-1)(delayed(optimize_single_instance)(c[i], cov, gamma, num_item, num_iter, principal_lin) for i in range(total_data))
    X, mu = zip(*results)
    X = np.array(X)
    mu = np.array(mu)
    if verbose:
        print("Done !")

    # Découpage
    Z_train, Z_test = Z[:num_data_train], Z[num_data_train:]
    r_train, r_test = c[:num_data_train], c[num_data_train:]
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
    
    
    