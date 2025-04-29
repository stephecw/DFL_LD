from gen_data import gen_datafile

if __name__ == "__main__":
    # Paramètres du dataset
    num_data_train = 500
    num_data_test = 100
    num_feat = 200
    num_item = [30, 50, 100]
    dim = [5,10]
    param = [[10,100]]

    for P in param:
        d = P[0]
        n = P[1]
        # Génération du dataset d'entraînement
        gen_datafile(num_data_train, num_data_test, num_feat, n, d, verbose=True)
