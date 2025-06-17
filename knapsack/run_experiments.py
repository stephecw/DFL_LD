import torch
from torch import optim
import numpy as np

from pyepo.model.grb import knapsackModel
from pyepo.func.utlis import sumGammaDistribution

from knapsack.data_import import ImportDataset
from train import train_MSE, train_classic, train_LD, train_SG, test
from models_class import CustomMLP
from diff_methods import I_MLE, SPOPlus
from opti_X_mu_CPU import OptimizationBatchModel
from knapsack.solver import solver_X_knapsack

import argparse


# Define command line arguments
parser = argparse.ArgumentParser(description="Training script with specified dimensions.")
parser.add_argument('--id', type=int, default=0, help='ID of the experiment. Used to differentiate runs.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')

parser.add_argument("--diff", type=str, default="IMLE", help="Name of the DFL model to evaluate ('SPOPlus', 'IMLE')")
parser.add_argument("--method", type=str, default="cla", help="Name of the training method to evaluate (e.g., 'cla', 'LD', 'SG', 'MSE')")
parser.add_argument('--decomp', type=int, nargs="+", default=[0], help='List of decomposition to use.')
parser.add_argument('--loss', type=int, default=0, help='Loss function to use (0 with penaty terms, 1 without).')

parser.add_argument('--dim', type=int, default=10, help='Number of constraints.')
parser.add_argument('--n', type=int, default=50, help='Number of items.')
parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
parser.add_argument('--patience', type=int, default=5, help='Number of epochs with no improvement after which learning rate will be reduced.')

parser.add_argument('--checkpoints', type=int, nargs="+", default=[30, 60, 180, 300, 600, 1800, 3600, 7200], help='Checkpoint for saving')

parser.add_argument('--step_mu', type=int, default=0, help='Number of epochs between mu updates. (0 to skip)')
parser.add_argument('--n_iter_mu', type=int, default=0, help='Number of iterations for mu optimization. (0 to skip)')

parser.add_argument("--lambd", type=float, default=10, help="Interpolation parameter for IMLE")
parser.add_argument("--sigma", type=float, default=1., help="Noise parameter for IMLE")
parser.add_argument("--n_samples", type=int, default=1, help="Number of samples for IMLE")
parser.add_argument("--kappa", type=int, default=5, help="Parameter kappa for IMLE noise distribution")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Training on:", device)

### EXPERIMENT EXECUTION ###
args = parser.parse_args()
id = args.id
seed = 

method = args.method
# Problem dimensions
num_feat = 12

num_data_train = 200  # Training dataset size
num_data_eval = 100   # eval dataset size
num_data_test = 1000  # Test dataset size

dim = args.dim
num_item = args.n
decomposition = args.decomp
loss = args.loss

checkpoints = args.checkpoints
batch_size = 200
lr = args.lr
model_shape = [num_feat, num_item]
dropout = 0.2

schedulerType = "ReduceLROnPlateau"  # "StepLR", "ReduceLROnPlateau", "OneCycleLR", None
sched_arg = {'mode':'min',
            'factor':0.5,
            'patience':args.patience,
            'min_lr':1e-6}
diff_method_name = args.diff
diff_method_arg = {}
if diff_method_name == "IMLE":
    diff_method_arg = {'n_samples':args.n_samples, 
                       'sigma':args.sigma,
                       'lambd':args.lambd,
                       'distribution':sumGammaDistribution(args.kappa)
                       }

step_mu = args.step_mu
num_iter_mu = args.n_iter_mu



def run_train(id, model, jobtype, dim, keep, loss, num_feat, num_item, num_data_train, num_data_eval, num_data_test,
              batch_size, epochs, lr,
              schedulerType, sched_arg,
              diff_method_name=None, diff_method_arg=None,
              step_mu=None, num_iter_mu=None,
              test_model=False, verbose=False, wandbarg=None, time_limit=None, save_model=True):
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
        #wandb.login(key="")  # Replace with your API key
        run = wandb.init(mode="offline", **wandbarg)

    # Load training dataset
    if verbose:
        print(f"Loading train_{dim}_{keep-1}_{num_feat}_{num_item}_{num_data_train}.txt", flush=True)
    try:
        train_set = ImportDataset(f"knapsack/datasets/train_{dim}_{keep-1}_{num_feat}_{num_item}_{num_data_train}.txt")
    except FileNotFoundError:
        print(f"File not found.", flush=True)
        return

    if verbose:
        print(f"Loading eval_{dim}_{num_feat}_{num_item}_{num_data_eval}.txt", flush=True)

    try:
        eval_set = ImportDataset(f"knapsack/datasets/eval_{dim}_{num_feat}_{num_item}_{num_data_eval}.txt", base=True)
    except FileNotFoundError:
        print(f"File not found.", flush=True)
        return

    # Create dataloaders
    train_loader = train_set.get_dataloader(batch_size=batch_size, shuffle=True)
    eval_loader = eval_set.get_dataloader(batch_size=batch_size, shuffle=False)

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
        
    # Solveur to compute regret when evaluating
    eval_solver = knapsackModel(weights=weights, capacity=capacities)
    # Training
    if jobtype == "LD":
        # Differentiation method for backpropagation when training 
        if diff_method_name == "IMLE":
            diff_method = I_MLE(knapsackModel(weights[:keep], capacities[:keep]), device, **diff_method_arg)
        elif diff_method_name == "SPOPlus":
            diff_method = SPOPlus(knapsackModel(weights[:keep], capacities[:keep]), device, **diff_method_arg)
        if verbose:
            print("Training the model with LD bound as loss...", flush=True)

        best_relat_regret, best_epoch, epoch = train_LD(model, diff_method, eval_solver,
                                    train_loader, eval_loader, optimizer, scheduler, 
                                    epochs, time_limit, eval_freq=int(1e8),
                                    run=run, verbose=verbose)
    elif jobtype == "cla":
        # Differentiation method for backpropagation when training 
        if diff_method_name == "IMLE":
            diff_method = I_MLE(knapsackModel(weights, capacities), device, **diff_method_arg)
        elif diff_method_name == "SPOPlus":
            diff_method = SPOPlus(knapsackModel(weights, capacities), device, **diff_method_arg)
        if verbose:
            print("Training the model with regret as loss...", flush=True)
            
        best_relat_regret, best_epoch, epoch = train_classic(model, diff_method, eval_solver, 
                                            train_loader, eval_loader, optimizer, scheduler, 
                                            epochs, time_limit, eval_freq=int(1e8),
                                            run=run, verbose=verbose)

    elif jobtype == "SG":
        # Differentiation method for backpropagation when training 
        if diff_method_name == "IMLE":
            diff_method = I_MLE(knapsackModel(weights[:keep], capacities[:keep]), device, **diff_method_arg)
        elif diff_method_name == "SPOPlus":
            diff_method = SPOPlus(knapsackModel(weights[:keep], capacities[:keep]), device, **diff_method_arg)
        # Optimizer for mu
        solvers = []
        solvers = [solver_X_knapsack(weights[:keep], capacities[:keep])]
        solvers += [solver_X_knapsack(np.expand_dims(weights[i], axis=0), np.expand_dims(capacities[i], axis=0)) for i in range(keep, dim)]
        optimizer_mu = OptimizationBatchModel(solvers)
        
        mu_global0 = torch.ones(len(train_loader.dataset), dim - keep, num_item, device=device, dtype=torch.float32)

        if verbose:
            print("Training the model with dynamic mu and LD bound as loss...", flush=True)

        best_relat_regret, best_epoch, epoch = train_SG(model, loss, diff_method, eval_solver, 
                                    train_loader, eval_loader, optimizer, scheduler, 
                                    epochs, time_limit, eval_freq=int(1e8),
                                    step_mu=step_mu, num_iter_mu=num_iter_mu, optimizer_mu=optimizer_mu,
                                    mu_global0=mu_global0,
                                    run=run, verbose=verbose)
    elif jobtype == "MSE":
        if verbose:
            print("Training the model with MSE as loss", flush=True)
        best_relat_regret, best_epoch, epoch = train_MSE(model, eval_solver, 
                                        train_loader, eval_loader, optimizer, scheduler,
                                        epochs, time_limit, eval_freq=int(1e8),
                                        run=run, verbose=verbose)

    # Save model
    if save_model:
        if jobtype == "LD":
            if verbose:
                print(f"Saving the model to knapsack/models/{diff_method_name}_LD_{dim}_{keep}_{num_feat}_{num_item}_{num_data_train}.pth", flush=True)
            torch.save(model.state_dict(), f'knapsack/models/{diff_method_name}_LD_{dim}_{keep}_{num_feat}_{num_item}_{num_data_train}.pth')
        elif jobtype == "cla":
            if verbose:
                print(f"Saving the model to knapsack/models/{diff_method_name}_classic_{dim}_{keep}_{num_feat}_{num_item}_{num_data_train}.pth", flush=True)
            torch.save(model.state_dict(), f'knapsack/models/{diff_method_name}_classic_{dim}_{keep}_{num_feat}_{num_item}_{num_data_train}.pth')
        elif jobtype == "SG":
            if verbose:
                print(f"Saving the model to knapsack/models/{diff_method_name}_SG_{dim}_{keep}_{num_feat}_{num_item}_{num_data_train}.pth", flush=True)
            torch.save(model.state_dict(), f'knapsack/models/{diff_method_name}_SG_{dim}_{keep}_{num_feat}_{num_item}_{num_data_train}.pth')
        elif jobtype == "MSE":
            if verbose:
                print(f"Saving the model to knapsack/models/MSE_{dim}_{keep}_{num_feat}_{num_item}_{num_data_train}.pth")
            torch.save(model.state_dict(), f'knapsack/models/MSE_{dim}_{keep}_{num_feat}_{num_item}_{num_data_train}.pth')


    # Évaluer d’abord sur le set d’éval avec le modèle au meilleur epoch
    regrets_eval = test(model, eval_loader, eval_solver, device, run=None)
    mean_relat_eval = np.mean(regrets_eval)
    std_relat_eval = np.std(regrets_eval)

    if verbose:
        print(f"Loading knapsack/datasets/test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt")
    try:
        test_set = ImportDataset(f"knapsack/datasets/test_{dim}_{num_feat}_{num_item}_{num_data_test}.txt", base=True)
    except FileNotFoundError:
        print(f"File not found.")

    test_loader = test_set.get_dataloader(batch_size=batch_size, shuffle=False)
    # Test the model on the test set
    regrets_test = test(model, test_loader, eval_solver, device, run)
    mean_relat_test = np.mean(regrets_test)
    std_relat_test = np.std(regrets_test)

    with open("knapsack/results_deg8_4.txt", 'a') as f:
        line = f"{id};{time_limit};{dim};{num_feat};{num_item};{num_data_train};{jobtype};{step_mu};{num_iter_mu};{diff_method_name};{loss};{lr};{best_relat_regret};{best_epoch};{epoch};{args.patience};{mean_relat_test};{std_relat_test};"
        line += ";".join(str(regrets_test[j]) for j in range(num_data_test - 1)) + f";{regrets_test[-1]}\n"
        f.write(line)
    
    # End execution
    if run is not None:
        run.finish()
        
    print(best_relat_regret, best_epoch, epoch, flush=True)
    

print(f"Training using {method} with {diff_method_name} for {epochs} epochs ({tl} seconds max) on {dim} constraints and {num_item} items.", flush=True)
model = CustomMLP(model_shape, dropout=dropout).to(device)
wandbarg = {
        'entity': "hugoper-polytechnique-montr-al",
        'project': "DFL_LD",
        'dir': "./knapsack/",
        'name': f"knapsack_{diff_method_name}_{method}_{dim}_{keep}_{num_feat}_{num_item}_{num_data_train}",
        'group': f"meeting_06_16",
        'job_type': f"{diff_method_name}_{method}_{tl}",
        'config': {
            "architecture": model_shape,
            "dropout": dropout,
            "dataset_train": f"train_{dim}_{num_feat}_{num_item}_{num_data_train}.txt",
            "dataset_eval": f"eval_{dim}_{num_feat}_{num_item}_{num_data_eval}.txt",
            "batch_size": batch_size,
            "epochs": epochs,
            "time_limit": tl,
            "learning_rate": lr,
            "schedulerType": schedulerType,
            "sched_arg": sched_arg,
            "diff_method": diff_method_name,
            "diff_method_arg": diff_method_arg,
            "step_mu": step_mu,
            "num_iter_mu": num_iter_mu,
            "num_feat": num_feat,
            "num_item": num_item,
            "num_data_train": num_data_train,
            "num_data_eval": num_data_eval,
            "num_data_test": num_data_test,
            "dim": dim,
            "loss": loss,
            "keep": keep,
            "id": id,
        }
}
run_train(id, model, method, dim, keep, loss, num_feat, num_item, num_data_train, num_data_eval, num_data_test,
        batch_size=batch_size, epochs=epochs, lr=lr, time_limit=tl,
        schedulerType=schedulerType, sched_arg=sched_arg,
        step_mu=step_mu, num_iter_mu=num_iter_mu,
        diff_method_name=diff_method_name, diff_method_arg=diff_method_arg,
        test_model=True, verbose=True, wandbarg=None, save_model=False)
