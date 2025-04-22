from gen_data import gen_datafile

if __name__ == "__main__":
    # Paramètres du dataset
    num_data_train = 500
    num_data_test = 100
    num_feat = 200
    num_item = [30, 50, 100]
    dim = [5,10]

    for d in dim:
        for n in num_item:
            # Génération du dataset
            if d != 10 and n != 50:
                gen_datafile(
                    num_data_train=num_data_train,
                    num_data_test=num_data_test,
                    num_feat=num_feat,
                    num_item=n,
                    dim=d,
                    verbose=True
                )
    