import torch

from pyepo.model.grb import knapsackModel

from knapsack.data_import import ImportDataset
from train import test
from models_class import CustomMLP

import argparse

# Define command line arguments
parser = argparse.ArgumentParser(description="Training script with specified dimensions.")
parser.add_argument("--diff", type=str, default="IMLE", help="Name of the DFL model to evaluate ('SPOPlus', 'IMLE')")
parser.add_argument("--method", type=str, default="cla", help="Name of the training method to evaluate (e.g., 'cla', 'LD', 'SG', 'MSE')")

parser.add_argument('--dim', type=int, default=5, help='Number of constraints.')
parser.add_argument('--n', type=int, default=30, help='Number of items.')
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

parser.add_argument('--step_mu', type=int, default=0, help='Number of epochs between mu updates. (0 to skip)')
parser.add_argument('--n_iter_mu', type=int, default=0, help='Number of iterations for mu optimization. (0 to skip)')

parser.add_argument("--lambd", type=float, default=1., help="Interpolation parameter for IMLE")
parser.add_argument("--sigma", type=float, default=1., help="Noise parameter for IMLE")
parser.add_argument("--n_samples", type=int, default=1, help="Number of samples for IMLE")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Testing on:", device)

### EXPERIMENT EXECUTION ###
args = parser.parse_args()
method = args.method
# Problem dimensions
num_feat = 200
num_data_train = 200  # Training dataset size
num_data_eval = 200   # eval dataset size
num_data_test = 1000

dim = args.dim
num_item = args.n

epochs = args.ep if args.ep > 0 else int(1e10)
tl = args.tl if args.tl > 0 else int(1e10)
batch_size = 32
lr = args.lr
model_shape = [num_feat, 100, num_item]
dropout = 0.2

schedulerType = "ReduceLROnPlateau"  # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg = {'mode':'min',
            'factor':0.5,
            'patience':40,
            'min_lr':1e-6}
diff_method_name = args.diff
diff_method_arg = {}
if diff_method_name == "IMLE":
    diff_method_arg = {'n_samples':args.n_samples, 
                       'sigma':args.sigma,
                       'lambd':args.lambd,
                       }

step_mu = args.step_mu
num_iter_mu = args.n_iter_mu

print(f"Loading test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt", flush=True)
try:
    test_set = ImportDataset(f"knapsack/datasets/test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt", test=True)
except FileNotFoundError:
    print(f"File not found.", flush=True)
    raise FileNotFoundError

model_shape = [num_feat, num_item]
dropout = 0.2
model = model = CustomMLP(model_shape, dropout=dropout).to(device)
file = f"knapsack/models/{method}"
if method != "MSE":
    file += f"_{diff_method_name}"
    for v in diff_method_arg.values():
        file += "_"+str(v).replace(".", "-")
file += f"_{dim}_{num_feat}_{num_item}_{lr}_{num_data_train}_{num_data_eval}"
if method == "SG":
    file += f"_{step_mu}_{num_iter_mu}"
file += f"_{schedulerType}"
for v in sched_arg.values():
    file += "_"+str(v).replace(".", "-")
file += ".pth"

print(f"Loading the model from {file}", flush=True)
try:
    model.load_state_dict(torch.load(file))
except FileNotFoundError:
    print(f"Model not found.", flush=True)
    raise FileNotFoundError

test_loader = test_set.get_dataloader(batch_size=batch_size, shuffle=False)
weights = test_set.get_weights(tensor=True)
capacities = test_set.get_capacities(tensor=True)

# Solveur to compute regret when evaluating
eval_solver = knapsackModel(weights=weights, capacity=capacities)
    
with open("knapsack/test_results_mini.txt", mode = "a") as file:
    rel_regret = test(model, test_loader, eval_solver, device)
    line = f"{method};{diff_method_name};{diff_method_arg};{dim};{num_feat};{num_item};{num_data_train};{lr};"
    line += f"{step_mu};{num_iter_mu};{schedulerType};{sched_arg};"
    line += ";".join(str(rel_regret[j]) for j in range(num_data_test - 1)) + f";{rel_regret[-1]}\n"
    file.write(line)