import torch
from torch import optim

from pyepo.model.grb import portfolioModel

from portfolio.data_import import ImportDataset
from train import train_MSE, train_classic, train_LD, train_SG, test
from models_class import CustomMLP
from diff_methods import I_MLE, SPOPlus
from opti_X_mu import OptimizationBatchModel
from portfolio.my_solver import BatchSolverLin, BatchSolverQuad, Solveur_lin, Solveur_quad, gb_portfolio_solver

import argparse

# Define command line arguments
parser = argparse.ArgumentParser(description="Training script with specified dimensions.")
parser.add_argument('--n', type=int, default=30, help='Number of items.')
parser.add_argument('--ep_cla', type=int, default=0, help='Number of epochs for classic training. (0 to skip)')
parser.add_argument('--ep_ld', type=int, default=0, help='Number of epochs for LD training. (0 to skip)')
parser.add_argument('--ep_sg', type=int, default=0, help='Number of epochs for SG training. (0 to skip)')
parser.add_argument('--ep_mse', type=int, default=0, help='Number of epochs for MSE training. (0 to skip)')
parser.add_argument('--step_mu', type=int, default=5, help='Number of epochs between mu updates. ')
parser.add_argument('--n_iter_mu', type=int, default=10, help='Number of iterations for mu optimization. ')
parser.add_argument('--lin', type= int, default=0, help='0 for the quadratic constraint and 1 for the linear one.')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Training on:", device)

def run_train(model, jobtype, gamma, num_feat, num_item, num_data_train, num_data_eval,num_data_test, principal_lin,
              batch_size, epochs, lr,
              schedulerType, sched_arg,
              diff_method_name=None, diff_method_arg=None,
              step_mu=5, num_iter_mu=15,
              verbose=False, wandbarg=None, time_limit=None, save_model=True):
    """
    Main function to load dataset and train the model.
    model: nn.Module: Model to train.
    LD: bool: If True, use Lagrangian decomposition.
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
    gamma_str = str(gamma).replace('.', '-')
    fold = "/lin" if principal_lin else "/quad"
    if verbose:
        print(f"Loading {fold}/train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt")
    try:
        train_set = ImportDataset(f"portfolio/datasets{fold}/train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt")
    except FileNotFoundError:
        print(f"File not found.")
        return

    if verbose:
        print(f"Loading {fold}/eval_{num_item}_{num_data_eval}_{num_feat}_{gamma_str}.txt")
    try:
        eval_set = ImportDataset(f"portfolio/datasets{fold}/eval_{num_item}_{num_data_eval}_{num_feat}_{gamma_str}.txt")
    except FileNotFoundError:
        print(f"File not found.")
        return
    
    if verbose:
        print(f"Loading {fold}/test_{num_item}_{num_data_test}_{num_feat}_{gamma_str}.txt")
    try:
        test_set = ImportDataset(f"portfolio/datasets{fold}/test_{num_item}_{num_data_test}_{num_feat}_{gamma_str}.txt")
    except FileNotFoundError:
        print(f"File not found.")
        return

    # Create dataloaders
    train_loader = train_set.get_dataloader(batch_size=batch_size, shuffle=True)
    eval_loader = eval_set.get_dataloader(batch_size=batch_size, shuffle=False)
    test_loader = test_set.get_dataloader(batch_size=batch_size, shuffle=False)

    # Problem parameters
    cov = 1e5*train_set.get_cov()

    # Model, optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr)
    scheduler = None
    if schedulerType == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(optimizer, **sched_arg)
    elif schedulerType == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_arg)
    elif schedulerType == "OneCycleLR":
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **sched_arg)
        
    # Solveur to compute regret when evaling
    eval_solver = portfolioModel(num_assets=num_item, covariance=cov, gamma=gamma)

    # Training
    if jobtype == "LD":
        # Differentiation method for backpropagation when training 
        if diff_method_name == "IMLE":
            solver = Solveur_lin(cov.shape[0], maximize=True) if principal_lin else Solveur_quad(cov.shape[0], cov, gamma, maximize=True)
            diff_method = I_MLE(solver, device, **diff_method_arg)
        elif diff_method_name == "SPOPlus":
            solver = Solveur_lin(cov.shape[0], maximize=False) if principal_lin else Solveur_quad(cov.shape[0], cov, gamma, maximize=False)
            diff_method = SPOPlus(solver, device, **diff_method_arg)
        if verbose:
            print("Training the model with LD bound as loss...")

        train_LD(model, diff_method, eval_solver,
                    train_loader, eval_loader, optimizer, scheduler, 
                    epochs, time_limit, eval_freq=1,
                    run=run, verbose=verbose)
    elif jobtype == "classic":
        # Differentiation method for backpropagation when training 
        if diff_method_name == "IMLE":
            solver = portfolioModel(num_assets=cov.shape[0], covariance=cov, gamma=gamma) 
            diff_method = I_MLE(solver, device, **diff_method_arg)
        elif diff_method_name == "SPOPlus":
            solver = gb_portfolio_solver(n_stocks = cov.shape[0], cov = cov, gamma = gamma, maximize=False)
            diff_method = SPOPlus(solver, device, **diff_method_arg)
        if verbose:
            print("Training the model with regret as loss...")
            
        train_classic(model, diff_method, eval_solver, 
                        train_loader, eval_loader, optimizer, scheduler, 
                        epochs, time_limit, eval_freq=1,
                        run=run, verbose=verbose)

    elif jobtype == "SG":
        # Differentiation method for backpropagation when training 
        if diff_method_name == "IMLE":
            solver = Solveur_lin(cov.shape[0], maximize=True) if principal_lin else Solveur_quad(cov.shape[0], cov, gamma, maximize=True)
            diff_method = I_MLE(solver, device, **diff_method_arg)
        elif diff_method_name == "SPOPlus":
            solver = Solveur_lin(cov.shape[0], maximize=False) if principal_lin else Solveur_quad(cov.shape[0], cov, gamma, maximize=False)
            diff_method = SPOPlus(solver, device, **diff_method_arg)
        # Optimizer for mu
        lin_solver = BatchSolverLin(num_item, device)
        quad_solver = BatchSolverQuad(num_item, cov, gamma, device)
        if principal_lin:
            solvers = [lin_solver, quad_solver]
        else :
            solvers = [quad_solver, lin_solver]
        optimizer_mu = OptimizationBatchModel(solvers, device)

        mu_global0 = torch.ones(len(train_loader.dataset), 1, num_item, device=device, dtype=torch.float32)

        if verbose:
            print("Training the model with dynamic mu and LD bound as loss...")

        train_SG(model, diff_method, eval_solver, 
                    train_loader, eval_loader, optimizer, scheduler, 
                    epochs, time_limit, eval_freq=1,
                    step_mu=step_mu, num_iter_mu=num_iter_mu, optimizer_mu=optimizer_mu,
                    mu_global0=mu_global0,
                    run=run, verbose=verbose)
    elif jobtype == "MSE":
        if verbose:
            print("Training the model with MSE as loss")
        train_MSE(model, eval_solver, 
                    train_loader, eval_loader, optimizer, scheduler,
                    epochs, time_limit, eval_freq=1,
                    run=run, verbose=verbose)
        
    # Test the model on the test set
    mean_relat_test, std_relat_test = test(model, test_loader, eval_solver, device, run)
    
    # Save model
    if save_model:
        if jobtype == "LD":
            if verbose:
                print(f"Saving the model to portfolio/models/{diff_method_name}_LD_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth")
            torch.save(model.state_dict(), f'portfolio/models/{diff_method_name}_LD_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth')
        elif jobtype == "classic":
            if verbose:
                print(f"Saving the model to portfolio/models/{diff_method_name}_classic_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth")
            torch.save(model.state_dict(), f'portfolio/models/{diff_method_name}_classic_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth')
        elif jobtype == "SG":
            if verbose:
                print(f"Saving the model to portfolio/models/{diff_method_name}_SG_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth")
            torch.save(model.state_dict(), f'portfolio/models/{diff_method_name}_SG_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth')
        elif jobtype == "MSE":
            if verbose:
                print(f"Saving the model to portfolio/models/{diff_method_name}_MSE_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth")
            torch.save(model.state_dict(), f'portfolio/models/{diff_method_name}_MSE_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.pth')


    # End execution
    if run is not None:
        run.finish()

### EXPERIMENT EXECUTION ###
args = parser.parse_args()

# Problem dimensions
num_feat = 200
num_data_train = 500  # Training dataset size
num_data_eval = 100   # eval dataset size
num_data_test = 200
gamma = 2.25
gamma_str = str(gamma).replace('.', '-')
num_item = args.n


# Classic parameters
epochs_classic = args.ep_cla
time_limit_classic = 600
batch_size_classic = 32
lr_classic = 0.002
model_shape_classic = [num_feat, 100, num_item]
dropout_classic = 0.2
schedulerType_classic = "StepLR"  # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg_classic = {'step_size':100,
                     'gamma':0.5
                     }
diff_method_classic = "SPOPlus"  # "StepLR", "SPOPlus"
diff_method_arg_classic = { }

# LD parameters
epochs_LD = args.ep_ld
time_limit_LD = 600
batch_size_LD = 32
lr_LD = 0.002
model_shape_LD = [num_feat, 100, num_item]
dropout_LD = 0.2
schedulerType_LD = "StepLR"  # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg_LD = {'step_size':100,
                'gamma':0.5
                }
diff_method_LD = "SPOPlus"  # "IMLE", "SPOPlus"
diff_method_arg_LD = { }
principal_lin = False if args.lin == 0 else True

# SG parameters
epochs_SG = args.ep_sg
time_limit_SG = 600
batch_size_SG = 32
lr_SG = 0.002
model_shape_SG = [num_feat, 100, num_item]
dropout_SG = 0.2
schedulerType_SG = "StepLR"  # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg_SG = {'step_size':100,
                'gamma':0.5
                }
diff_method_SG = "SPOPlus"  # "IMLE", "SPOPlus"
diff_method_arg_SG = {}
step_mu = args.step_mu
num_iter_mu = args.n_iter_mu

# MSE parameters
epochs_MSE = args.ep_mse
time_limit_MSE = 3600
batch_size_MSE = 32
lr_MSE = 0.001
model_shape_MSE = [num_feat, 100, num_item]
dropout_MSE = 0.2
schedulerType_MSE = None  # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg_MSE = {'step_size':100,
                     'gamma':0.5
                     }


print(f"Training for {epochs_classic} epochs for classic model, {epochs_LD} epochs for LD model, {epochs_SG} for SG model with mu and {epochs_MSE} for MSE model on {num_item} items.")


### EXECUTION ###
## LD ##
if epochs_LD > 0:
    model = CustomMLP(model_shape_LD, dropout=dropout_LD).to(device)
    wandbarg = {
            'entity': "hugoper-polytechnique-montr-al",
            'project': "DFL_LD",
            'dir': "./",
            'name': f"{diff_method_LD}_LD_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'group': f"portfolio_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'job_type': "portfolio_LD",
            'principal': "linear" if principal_lin else "quad",
            'config': {
                "architecture": model_shape_LD,
                "dropout": dropout_LD,
                "dataset_train": f"train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "dataset_eval": f"eval_{num_item}_{num_data_eval}_{num_feat}_{gamma_str}.txt",
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
    run_train(model, "LD", gamma, num_feat, num_item, num_data_train, num_data_eval,
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
            'name': f"{diff_method_classic}_classic_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'group': f"portfolio_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'job_type': "portfolio_classic",
            'config': {
                "architecture": model_shape_classic,
                "dropout": dropout_classic,
                "dataset_train": f"train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "dataset_eval": f"eval_{num_item}_{num_data_eval}_{num_feat}_{gamma_str}.txt",
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
    run_train(model, "classic", gamma, num_feat, num_item, num_data_train, num_data_eval, num_data_test, principal_lin,
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
            'name': f"{diff_method_SG}_SG_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'group': f"portfolio_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'job_type': "portfolio_SG",
            'config': {
                "architecture": model_shape_SG,
                "dropout": dropout_SG,
                "dataset_train": f"train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "dataset_eval": f"eval_{num_item}_{num_data_eval}_{num_feat}_{gamma_str}.txt",
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
    run_train(model, "SG", gamma, num_feat, num_item, num_data_train, num_data_eval, num_data_test, principal_lin,
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
            'name': f"MSE_{num_item}_{num_data_train}_{num_feat}_{gamma_str}",
            'group': f"presentation_05_11",
            'job_type': "portfolio_MSE",
            'config': {
                "architecture": model_shape_MSE,
                "dropout": dropout_MSE,
                "dataset_train": f"train_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "dataset_eval": f"eval_{num_item}_{num_data_train}_{num_feat}_{gamma_str}.txt",
                "batch_size": batch_size_MSE,
                "epochs": epochs_MSE,
                "time_limit": time_limit_MSE,
                "learning_rate": lr_MSE,
                "schedulerType": schedulerType_MSE,
                "sched_arg": sched_arg_MSE,
            }
    }
    run_train(model, "MSE", gamma, num_feat, num_item, num_data_train, num_data_eval,num_data_test, principal_lin,
            batch_size=batch_size_MSE, epochs=epochs_MSE, lr=lr_MSE,
            schedulerType=schedulerType_MSE, sched_arg=sched_arg_MSE,
            verbose=True, wandbarg=wandbarg, time_limit=time_limit_MSE)
