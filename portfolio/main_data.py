from gen_data import gen_datafile

# Paramètres du dataset
num_data_train = 500
num_data_test = 100
num_feat = 200
num_iter = 300
num_item = [30, 50, 100]
gamma = 2.25 # Par défaut
for n in num_item:
    gen_datafile(num_data_train, num_data_test, num_feat, n, gamma, num_iter, verbose=True)
