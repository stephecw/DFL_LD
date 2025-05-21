import torch
from torch import optim

from pyepo.model.grb import knapsackModel

from data_tools.data_import import ImportDataset
from train import train_MSE, train_classic, train_LD, train_SG
from models_class import CustomMLP
from diff_methods import I_MLE, SPOPlus
from opti_X_mu import OptimizationBatchModel
from knapsack.solver import solver_X_1D_knapsack

import argparse

# Define command line arguments
parser = argparse.ArgumentParser(description="Training script with specified dimensions.")
parser.add_argument('--dim', type=int, default=5, help='Number of constraints.')
parser.add_argument('--n', type=int, default=30, help='Number of items.')
parser.add_argument('--ep_cla', type=int, default=0, help='Number of epochs for classic training. (0 to skip, -1 to use time limit)')
parser.add_argument('--tl_cla', type=int, default=0, help='Time limit for classic training. (0 for doing all epochs)')
parser.add_argument('--ep_ld', type=int, default=0, help='Number of epochs for LD training. (0 to skip, -1 to use time limit)')
parser.add_argument('--tl_ld', type=int, default=0, help='Time limit for LD training. (0 for doing all epochs)')
parser.add_argument('--ep_sg', type=int, default=0, help='Number of epochs for SG training. (0 to skip, -1 to use time limit)')
parser.add_argument('--tl_sg', type=int, default=0, help='Time limit for SG training. (0 for doing all epochs)')
parser.add_argument('--ep_mse', type=int, default=0, help='Number of epochs for MSE training. (0 to skip, -1 to use time limit)')
parser.add_argument('--tl_mse', type=int, default=0, help='Time limit for MSE training. (0 for doing all epochs)')
parser.add_argument('--step_mu', type=int, default=5, help='Number of epochs between mu updates. (0 to skip)')
parser.add_argument('--n_iter_mu', type=int, default=10, help='Number of iterations for mu optimization. (0 to skip)')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Training on:", device)

def run_train(model, jobtype, dim, num_feat, num_item, num_data_train, num_data_test,
              batch_size, epochs, lr,
              schedulerType, sched_arg,
              diff_method_name=None, diff_method_arg=None,
              step_mu=5, num_iter_mu=15,
              verbose=False, wandbarg=None, time_limit=None, save_model=True):
    """
    Main function to load dataset and train the model.
    model: nn.Module: Model to train.
    LD: bool: If True, use Lagrangian decomposition.
    dim: int: Number of dimensions.
    num_feat: int: Number of features.
    num_item: int: Number of items.
    num_data_train: int: Number of training data points.
    batch_size: int: Batch size.
    epochs: int: Number of training epochs.
    lr: float: Learning rate.
    schedulerType: str: Type of scheduler to use. 'StepLR' or 'None'.
    sched_step_size: int: Step size for scheduler.
    sched_gamma: float: Gamma for scheduler.
    IMLE_n_samples: int: Number of Monte Carlo samples for IMLE.
    IMLE_sigma: float: Sigma for IMLE.
    IMLE_lambd: float: Lambda for IMLE.
    IMLE_two_sides: bool: If True, use two-sided perturbation.
    IMLE_processes: int: Number of processes for IMLE.
    verbose: bool: If True, show progress.
    wandbarg: dict: Arguments for wandb.init() if wandb is used, else None.
    save_model: bool: If True, save model after training.
    """
    run = None
    if wandbarg is not None:
        import wandb
        #wandb.login(key="c656dc47be1ed8b7866027b0569dca27b78821d9")  # Replace with your API key
        run = wandb.init(mode="offline", **wandbarg)

    # Load training dataset
    if verbose:
        print(f"Loading train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")
    try:
        train_set = ImportDataset(f"knapsack/datasets/train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt")

    except FileNotFoundError:
        print(f"File not found.")
        return

    if verbose:
        print(f"Loading eval_{dim}_{num_feat}_{num_item}_{num_data_test}.txt")
    try:
        test_set = ImportDataset(f"knapsack/datasets/eval_{dim}_{num_feat}_{num_item}_{num_data_test}.txt")

    except FileNotFoundError:
        print(f"File not found.")
        return

    # Create dataloaders
    train_loader = train_set.get_dataloader(batch_size=batch_size, shuffle=True)
    test_loader = test_set.get_dataloader(batch_size=batch_size, shuffle=False)

    # Problem parameters
    weights = train_set.get_weights(tensor=True)
    capacities = train_set.get_capacities(tensor=True)

    # Model, optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = None
    if schedulerType == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, **sched_arg)
    elif schedulerType == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_arg)
    elif schedulerType == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **sched_arg)
        
    # Solveur to compute regret when testing
    test_solver = knapsackModel(weights=weights, capacity=capacities)

    # Training
    if jobtype == "LD":
        # Differentiation method for backpropagation when training 
        if diff_method_name == "IMLE":
            diff_method = I_MLE(knapsackModel(weights[0].unsqueeze(0), capacities[0].unsqueeze(0)), device, **diff_method_arg)
        elif diff_method_name == "SPOPlus":
            diff_method = SPOPlus(knapsackModel(weights[0].unsqueeze(0), capacities[0].unsqueeze(0)), device, **diff_method_arg)
        if verbose:
            print("Training the model with LD bound as loss...")

        train_LD(model, diff_method, test_solver,
                    train_loader, test_loader, optimizer, scheduler, 
                    epochs, time_limit, eval_freq=1,
                    run=run, verbose=verbose)
    elif jobtype == "classic":
        # Differentiation method for backpropagation when training 
        if diff_method_name == "IMLE":
            diff_method = I_MLE(knapsackModel(weights, capacities), device, **diff_method_arg)
        elif diff_method_name == "SPOPlus":
            diff_method = SPOPlus(knapsackModel(weights, capacities), device, **diff_method_arg)
        if verbose:
            print("Training the model with regret as loss...")
            
        train_classic(model, diff_method, test_solver, 
                        train_loader, test_loader, optimizer, scheduler, 
                        epochs, time_limit, eval_freq=1,
                        run=run, verbose=verbose)

    elif jobtype == "SG":
        # Differentiation method for backpropagation when training 
        if diff_method_name == "IMLE":
            diff_method = I_MLE(knapsackModel(weights[0].unsqueeze(0), capacities[0].unsqueeze(0)), device, **diff_method_arg)
        elif diff_method_name == "SPOPlus":
            diff_method = SPOPlus(knapsackModel(weights[0].unsqueeze(0), capacities[0].unsqueeze(0)), device, **diff_method_arg)
        # Optimizer for mu
        solvers = [solver_X_1D_knapsack(weights[i], capacities[i], device) for i in range(dim)]
        optimizer_mu = OptimizationBatchModel(solvers, device)

        if verbose:
            print("Training the model with dynamic mu and LD bound as loss...")

        train_SG(model, diff_method, test_solver, 
                    train_loader, test_loader, optimizer, scheduler, 
                    epochs, time_limit, eval_freq=1,
                    step_mu=step_mu, num_iter_mu=num_iter_mu, optimizer_mu=optimizer_mu,
                    num_items=num_item, dim=dim,
                    run=run, verbose=verbose)
    elif jobtype == "MSE":
        if verbose:
            print("Training the model with MSE as loss")
        train_MSE(model, test_solver, 
                    train_loader, test_loader, optimizer, scheduler,
                    epochs, time_limit, eval_freq=1,
                    run=run, verbose=verbose)

    # Save model
    if save_model:
        if jobtype == "LD":
            if verbose:
                print(f"Saving the model to knapsack/models/{diff_method_name}_LD_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'knapsack/models/{diff_method_name}_LD_{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
        elif jobtype == "classic":
            if verbose:
                print(f"Saving the model to knapsack/models/{diff_method_name}_classic_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'knapsack/models/{diff_method_name}_classic_{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
        elif jobtype == "SG":
            if verbose:
                print(f"Saving the model to knapsack/models/{diff_method_name}_SG_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'knapsack/models/{diff_method_name}_SG_{dim}_{num_feat}_{num_item}_{num_data_train}.pth')
        elif jobtype == "MSE":
            if verbose:
                print(f"Saving the model to knapsack/models/MSE_{dim}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'knapsack/models/MSE_{dim}_{num_feat}_{num_item}_{num_data_train}.pth')


    # End execution
    if run is not None:
        run.finish()

### EXPERIMENT EXECUTION ###
args = parser.parse_args()

# Problem dimensions
num_feat = 200
num_data_train = 500  # Training dataset size
num_data_test = 100   # Test dataset size
dim = args.dim
num_item = args.n

# Classic parameters
epochs_classic = args.ep_cla if args.ep_cla >= 0 else int(1e10)
time_limit_classic = args.tl_cla if args.tl_cla > 0 else int(1e10)
batch_size_classic = 32
lr_classic = 0.001
model_shape_classic = [num_feat, 100, num_item]
dropout_classic = 0.2
schedulerType_classic = None  # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg_classic = {'step_size':100,
                     'gamma':0.5
                     }
diff_method_classic = "IMLE"  # "StepLR", "SPOPlus"
diff_method_arg_classic = { }

# LD parameters
epochs_LD = args.ep_ld if args.ep_ld >= 0 else int(1e10)
time_limit_LD = args.tl_ld if args.tl_ld > 0 else int(1e10)
batch_size_LD = 32
lr_LD = 0.001
model_shape_LD = [num_feat, 100, num_item]
dropout_LD = 0.2
schedulerType_LD = None  # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg_LD = {'step_size':100,
                     'gamma':0.5
                     }
diff_method_LD = "IMLE"  # "IMLE", "SPOPlus"
diff_method_arg_LD = { }

# SG parameters
epochs_SG = args.ep_sg if args.ep_sg >= 0 else int(1e10)
time_limit_SG = args.tl_sg if args.tl_sg > 0 else int(1e10)
batch_size_SG = 32
lr_SG = 0.001
model_shape_SG = [num_feat, 100, num_item]
dropout_SG = 0.2
schedulerType_SG = None  # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg_SG = {'step_size':100,
                     'gamma':0.5
                     }
diff_method_SG = "IMLE"  # "IMLE", "SPOPlus"
diff_method_arg_SG = {}
step_mu = args.step_mu
num_iter_mu = args.n_iter_mu

# MSE parameters
epochs_MSE = args.ep_mse if args.ep_mse >= 0 else int(1e10)
time_limit_MSE = args.tl_mse if args.tl_mse > 0 else int(1e10)
batch_size_MSE = 32
lr_MSE = 0.001
model_shape_MSE = [num_feat, 100, num_item]
dropout_MSE = 0.2
schedulerType_MSE = None  # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg_MSE = {'step_size':100,
                     'gamma':0.5
                     }


print(f"Training for {epochs_classic} epochs for classic model, {epochs_LD} epochs for LD model, {epochs_SG} for SG model with mu and {epochs_MSE} for MSE model on {dim} constraints and {num_item} items.")


### EXECUTION ###
## LD ##
if epochs_LD > 0:
    model = CustomMLP(model_shape_LD, dropout=dropout_LD).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"{diff_method_LD}_LD_{dim}_{num_feat}_{num_item}_{num_data_train}",
            'group': f"presentation_05_26",
            'job_type': "LD",
            'config': {
                "architecture": model_shape_LD,
                "dropout": dropout_LD,
                "dataset_train": f"train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt",
                "dataset_test": f"test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt",
                "batch_size": batch_size_LD,
                "epochs": epochs_LD,
                "time_limit": time_limit_LD,
                "learning_rate": lr_LD,
                "schedulerType": schedulerType_LD,
                "sched_arg": sched_arg_LD,
                "diff_method": diff_method_LD,
                "diff_method_arg": diff_method_arg_LD
            }
    }
    run_train(model, "LD", dim, num_feat, num_item, num_data_train, num_data_test,
            batch_size=batch_size_LD, epochs=epochs_LD, lr=lr_LD,
            schedulerType=schedulerType_LD, sched_arg=sched_arg_LD,
            diff_method_name=diff_method_LD, diff_method_arg=diff_method_arg_LD,
            verbose=True, wandbarg=wandbarg, time_limit=time_limit_LD)

## Classic ##
if epochs_classic > 0:
    model = CustomMLP(model_shape_classic, dropout=dropout_classic).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"{diff_method_classic}_Classic_{dim}_{num_feat}_{num_item}_{num_data_train}",
            'group': f"presentation_05_26",
            'job_type': "Classic",
            'config': {
                "architecture": model_shape_classic,
                "dropout": dropout_classic,
                "dataset_train": f"train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt",
                "dataset_test": f"test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt",
                "batch_size": batch_size_classic,
                "epochs": epochs_classic,
                "time_limit": time_limit_classic,
                "learning_rate": lr_classic,
                "schedulerType": schedulerType_classic,
                "sched_arg": sched_arg_classic,
                "diff_method": diff_method_classic,
                "diff_method_arg": diff_method_arg_classic
            }
    }
    run_train(model, "classic", dim, num_feat, num_item, num_data_train, num_data_test,
            batch_size=batch_size_classic, epochs=epochs_classic, lr=lr_classic,
            schedulerType=schedulerType_classic, sched_arg=sched_arg_classic,
            diff_method_name=diff_method_classic, diff_method_arg=diff_method_arg_classic,
            verbose=True, wandbarg=wandbarg, time_limit=time_limit_classic)

## SG ##
if epochs_SG > 0:
    model = CustomMLP(model_shape_SG, dropout=dropout_SG).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"{diff_method_SG}_SG_{dim}_{num_feat}_{num_item}_{num_data_train}",
            'group': f"presentation_05_26",
            'job_type': "SG",
            'config': {
                "architecture": model_shape_SG,
                "dropout": dropout_SG,
                "dataset_train": f"train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt",
                "dataset_test": f"test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt",
                "batch_size": batch_size_SG,
                "epochs": epochs_SG,
                "time_limit": time_limit_SG,
                "learning_rate": lr_SG,
                "schedulerType": schedulerType_SG,
                "sched_arg": sched_arg_SG,
                "diff_method": diff_method_SG,
                "diff_method_arg": diff_method_arg_SG,
                "step_mu": step_mu,
                "num_iter_mu": num_iter_mu
            }
    }
    run_train(model, "SG", dim, num_feat, num_item, num_data_train, num_data_test,
            batch_size=batch_size_SG, epochs=epochs_SG, lr=lr_SG,
            schedulerType=schedulerType_SG, sched_arg=sched_arg_SG,
            diff_method_name=diff_method_SG, diff_method_arg=diff_method_arg_SG,
            step_mu=step_mu, num_iter_mu=num_iter_mu,
            verbose=True, wandbarg=wandbarg, time_limit=time_limit_SG)

if epochs_MSE > 0:
    model = CustomMLP(model_shape_MSE, dropout=dropout_MSE).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"{diff_method_SG}_MSE_{dim}_{num_feat}_{num_item}_{num_data_train}",
            'group': f"presentation_05_26",
            'job_type': "MSE",
            'config': {
                "architecture": model_shape_MSE,
                "dropout": dropout_MSE,
                "dataset_train": f"train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt",
                "dataset_test": f"test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt",
                "batch_size": batch_size_MSE,
                "epochs": epochs_MSE,
                "time_limit": time_limit_MSE,
                "learning_rate": lr_MSE,
                "schedulerType": schedulerType_MSE,
                "sched_arg": sched_arg_MSE,
            }
    }
    run_train(model, "MSE", dim, num_feat, num_item, num_data_train, num_data_test,
            batch_size=batch_size_MSE, epochs=epochs_MSE, lr=lr_MSE,
            schedulerType=schedulerType_MSE, sched_arg=sched_arg_MSE,
            verbose=True, wandbarg=wandbarg, time_limit=time_limit_MSE)
