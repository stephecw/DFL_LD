from gen_data import gen_datafile
import argparse

# Définir les arguments de ligne de commande
parser = argparse.ArgumentParser(description="Script de génération de dataset avec des dimensions spécifiées.")
parser.add_argument('--n', type=int, default=50, help='Nombre d\'item.')
parser.add_argument('--gamma', type=float, default=2.25, help='Gamma.')
parser.add_argument('--n_train', type=int, default=500, help='Nombre de données d\'entrainement')
parser.add_argument('--n_test', type=int, default=100, help='Nombre de données de test')
parser.add_argument('--n_feat', type=int, default=200, help='Nombre de features')
parser.add_argument('--lin', type=int, default=1, help='1 pour prendre la contrainte linéraire pour le sous-prob principal, 0 pour la contrainte quadratique')
parser.add_argument('--n_iter', type=int, default=300, help='Nombre d\'itérations pour l\'optimisation de \mu. (0 pour ne pas l\'exécuter)')


# Paramètres du dataset
args = parser.parse_args()
num_data_train = args.n_train
num_data_test = args.n_test
num_feat = args.n_feat
num_iter = args.n_iter
num_item = args.n
gamma = args.gamma
principal_lin = False if args.lin == 0 else True
gen_datafile(num_data_train, num_data_test, num_feat, num_item, gamma, num_iter, principal_lin = principal_lin, verbose=True)

