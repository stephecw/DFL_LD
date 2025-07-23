import numpy as np
import torch
import pyepo.data as data
from pyepo.model.grb import portfolioModel
from portfolio.my_solver import BatchSolverLin, BatchSolverQuad, BatchSolverExact
from opti_X_mu_CPU import OptimizationBatchModel
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def write_dataset_file(fname, num_feat, num_item, num_data, deg, cov, gamma, Z, c, x_star_array, X, mu):
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
    total_data = num_data_train + num_data_val + num_data_test 

    if verbose:
        print(f"➡ Génération de {total_data} instances ({num_data_train} train, {num_data_test} test)")
        print(f"➡ Dimensions : {num_item} items, {num_feat} features")
        print(f"➡ Degré du polynôme : {deg}")

    # Données aléatoires
    cov, Z, c = data.portfolio.genData(total_data, num_feat, num_item, deg=deg, noise_level=1, seed=135)
    gamma = gam  # risk_level = gamma * mean(cov[i])
    cov2 = 1e5*cov  # Covariance pour la contrainte quadratique
    

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
    # Optimisation de mu et X°
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

    # Découpage
    Z_train, Z_val, Z_test = Z[:num_data_train], Z[num_data_train:num_data_train + num_data_val], Z[num_data_train + num_data_val:]
    c_train, c_val, c_test = c[:num_data_train], c[num_data_train:num_data_train + num_data_val], c[num_data_train + num_data_val:]
    x_star_train, x_star_val, x_star_test = x_star_array[:num_data_train], x_star_array[num_data_train:num_data_train + num_data_val], x_star_array[num_data_train + num_data_val:]
    X_train = X
    mu_train = mu
    X_val = np.zeros((num_data_val, num_item), dtype=float)
    mu_val = np.zeros((num_data_val, num_item), dtype=float)
    X_test = np.zeros((num_data_test, num_item), dtype=float)
    mu_test = np.zeros((num_data_test, num_item), dtype=float)
    
    # print(f"x : {x_star_train[0]}")
    # print(f"X : {X_train[0]}")
    # print(f"mu : {mu_train[0]}")
    # margin = np.array([gamma*np.mean(cov) - np.dot(X[i].T, np.dot(cov, X[i])) for i in range(total_data)])
    # print(margin)
    # print(margin.mean())
    # print(margin.std())
    # print()
    # print(gamma*np.mean(cov))

    # Sauvegarde
    gamma_str = str(gamma).replace('.', '-')
    fold = "/lin" if principal_lin else "/quad"
    write_dataset_file(f"portfolio/datasets/train_{num_item}_{num_data_train}_{num_feat}_{deg}_{gamma_str}.txt",
                       num_feat, num_item, num_data_train, deg,
                       cov, gamma, Z_train, c_train, x_star_train, X_train, mu_train)

    write_dataset_file(f"portfolio/datasets/validation_{num_item}_{num_data_val}_{num_feat}_{deg}_{gamma_str}.txt",
                       num_feat, num_item, num_data_val, deg,
                       cov, gamma, Z_val, c_val, x_star_val, X_val, mu_val)
    
    write_dataset_file(f"portfolio/datasets/test_{num_item}_{num_data_test}_{num_feat}_{deg}_{gamma_str}.txt",
                       num_feat, num_item, num_data_test, deg,
                       cov, gamma, Z_test, c_test, x_star_test, X_test, mu_test)
    
if __name__ == "__main__":

    # Définir les arguments de ligne de commande
    parser = argparse.ArgumentParser(description="Script de génération de dataset avec des dimensions spécifiées.")
    parser.add_argument('--n', type=int, default=50, help='Nombre d\'item.')
    parser.add_argument('--gamma', type=float, default=2.25, help='Gamma.')
    parser.add_argument('--n_train', type=int, default=100, help='Nombre de données d\'entrainement')
    parser.add_argument('--n_validation', type=int, default=25, help='Nombre de données de validation')
    parser.add_argument('--n_test', type=int, default=10000, help='Nombre de données de test')
    parser.add_argument('--n_feat', type=int, default=5, help='Nombre de features')
    parser.add_argument('--lin', type=int, default=0, help='1 pour prendre la contrainte linéraire pour le sous-prob principal, 0 pour la contrainte quadratique')
    parser.add_argument('--n_iter', type=int, default=500, help='Nombre d\'itérations pour l\'optimisation de \mu. (0 pour ne pas l\'exécuter)')
    parser.add_argument('--deg', type=int, default=8, help='Degré du polynôme pour la génération des données')


    # Paramètres du dataset
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
    
    