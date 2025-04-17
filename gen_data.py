import numpy as np
import pyepo
import pyepo.data as data

def gen_datafile(num_data, num_feat, num_item, dim, fname=None):
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
    cap_2, poid_2_1, poid_2_2, ..., poid_2_{num_item}
    ... 
    cap_{dim}, poid_{dim}_1, ..., poid__{dim}_{num_item}
    z_1_1, z_1_2, ..., z_1_{num_feat}, c_1_1, c_1_2, ..., c_1_{num_item}
    z_2_1, z_2_2, ..., z_2_{num_feat}, c_2_1, c_2_2, ..., c_2_{num_item}
    ...
    z_{num_data}_1, ..., z_{num_data}_{num_feat}, c_{num_data}_1, ..., c_{num_data}_{num_item}
    '''
    weights, x, c = pyepo.data.knapsack.genData(num_data, num_feat, num_item, dim, deg=4, noise_width=0, seed=135)
    capacities = np.random.random()*0.2+0.2*np.sum(weights,axis=1)
    if fname is None:
        fname = f"datasets/train_{dim}_{num_feat}_{num_item}_{num_data}.txt"
    with open(fname,'x') as f:
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
                line+= str(x[i][j]) + ","
            for j in range(num_item-1):
                line+= str(int(c[i][j])) + ","
            line += str(int(c[i][-1])) + "\n"
            f.write(line)


num_data = 1000 # Taille du dataset
num_feat = 20 # Nombre de features en entrée du NN
num_item = 30 # Nombre d'items
dim = 5 # Nombre de contraintes

gen_datafile(num_data, num_feat, num_item, dim) # Génération du fichier de données