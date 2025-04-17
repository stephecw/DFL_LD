import numpy as np
import pyepo
import pyepo.data as data
from pyepo.model.grb import knapsackModel
import gurobipy as gp
from opti_X_mu import OptimizationModel
def f(x,c):
    return np.sum(c*x)

def find_X_mu(num_item, dim, c, weights, capacities, verbose=False):
    """
    Trouve la valeur de X°_1 et mu pour le problème du sac à dos multi-dimensionnel.
    c : np.array : Coûts des items
    num_item : int : Nombre d'items
    dim : int : Nombre de contraintes
    """
    optimizer = OptimizationModel(num_item, dim, c, weights, capacities, f)
    optimizer.optim_mu(verbose, max_iter=10000)
    mu = optimizer.get_mu().flatten()
    X0 = optimizer.get_X0()
    X = optimizer.get_X()
    print(X)
    return X0, mu
    

def gen_datafile(num_data, num_feat, num_item, dim, fname=None, verbose=False):
    '''
    Génération d'un fichier de données pour le problème du sac à dos multi-dimensionnel.
    num_data : int : Nombre de données à générer
    num_feat : int : Nombre de features en entrée du NN
    num_item : int : Nombre d'items
    dim : int : Nombre de contraintes
    fname : str : Nom du fichier de données à générer
    
    Les poids et capacités sont identiques pour chaque données du dataset.
    Forme du fichier de données :
    dim,num_feat,num_item,num_data
    cap_1, poid_1_1, poid_1_2, ..., poid_1_{num_item}
    ... 
    cap_{dim}, poid_{dim}_1, ..., poid__{dim}_{num_item}
    Z_1_1, ..., Z_1_{num_feat}, c_1_1, ..., c_1_{num_item}, X_1_1, ..., X_1_{num_item}, mu_1_1, ..., mu_1_{num_item*(dim-1)}
    ...
    Z_{num_data}_1, ..., Z_{num_data}_{num_feat}, c_{num_data}_1, ..., c_{num_data}_{num_item}, X_{num_data}_1, ..., X_{num_data}_{num_item}, mu_{num_data}_1, ..., mu_{num_data}_{num_item*(dim-1)}
    '''
    if verbose:
        print(f"Generating {num_data} data with {num_feat} features, {num_item} items and {dim} constraints :")
    weights, Z, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item, dim, deg=4, noise_width=0, seed=135)

    capacities = np.random.random()*0.1+0.2*np.sum(weights,axis=1)
  
    # Résolution exacte du problème pour chaque instance (x*)
    x_star_list = []
    for i in range(num_data):
        model = knapsackModel(weights=weights, capacity=capacities)
        model.setObj(c[i])
        x_star, _ = model.solve()
        x_star_list.append(x_star)

    x_star_array = np.array(x_star_list)
    
    if verbose:
        print("Start searching for optimal X and mu for the data :\n")
    X = np.zeros((num_data, num_item), dtype=int)
    mu = np.zeros((num_data, (dim-1) * num_item), dtype=float)
    for i in range(num_data):
        if verbose:
            print(f"    Data {i+1}/{num_data} :")
        X[i], mu[i] = find_X_mu(num_item, dim, c[i], weights, capacities, verbose)
    
    if fname is None:
        fname = f"datasets/train_{dim}_{num_feat}_{num_item}_{num_data}.txt"
    with open(fname,'w') as f:
        line = f"{dim},{num_feat},{num_item},{num_data}\n"
        f.write(line)
        for i in range(dim):
            line = str(int(capacities[i]))+","
            for j in range(num_item-1):
                line += str(int(weights[i][j]))+","
            line += str(int(weights[i][-1]))+"\n"
            f.write(line)
        for i in range(num_data):
            line = ""
            for j in range(num_feat):
                line+= str(Z[i][j]) + ","
            for j in range(num_item):
                line+= str(int(c[i][j])) + ","
            for j in range(num_item):
                line += str(int(x_star_array[i][j])) + ","
            for j in range(num_item):
                line+= str(int(X[i][j])) + ","
                line+= str(0) + ","
            for j in range(num_item*(dim-1)-1):   
                line+= str(mu[i][j]) + ","
            line += str(mu[i][-1]) + "\n" 
            f.write(line)



num_data = 500 # Taille du dataset
num_feat = 20 # Nombre de features en entrée du NN
num_item = 30 # Nombre d'items
dim = 5 # Nombre de contraintes

gen_datafile(num_data, num_feat, num_item, dim, None) # Génération du fichier de données