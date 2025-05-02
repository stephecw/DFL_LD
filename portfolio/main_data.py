from gen_data import gen_datafile

# Paramètres du dataset
num_data_train = 1
num_data_test = 0
num_feat = 5
num_iter = 500
num_item = [5]
gamma = 2.25 # Par défaut
for n in num_item:
    gen_datafile(num_data_train, num_data_test, num_feat, n, gamma, num_iter, principal_lin = False, verbose=True)
