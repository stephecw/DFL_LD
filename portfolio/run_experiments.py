import argparse, os, csv, time, torch, numpy as np
from torch import optim
from pyepo.model.grb import portfolioModel

from portfolio.data_import import ImportDataset
from train import train, test                     # new train()
from models_class import CustomMLP
from diff_methods import I_MLE, SPOPlus, Exact, MSE
from opti_X_mu_CPU import OptimizationBatchModel_serial
from portfolio.my_solver import (
    Solveur_lin, Solveur_quad, BatchSolverExact, BatchSolverLin, BatchSolverQuad
)
from utils import seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Training on:", device)

# ARGPARSE 
parser = argparse.ArgumentParser("Experiments with new train.py")
# dimensions & data
parser.add_argument('--n', type=int, default=50, help='Number of items')
parser.add_argument('--deg', type=int, default=8, help='Polynomial degree for feature generation')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--regenerate', type=int, default=0, help='Regenerate datasets')
# training choices
parser.add_argument('--ep_classic', type=int, default=0)
parser.add_argument('--ep_ld',      type=int, default=0)
parser.add_argument('--ep_sg',      type=int, default=0)
parser.add_argument('--ep_mse',     type=int, default=0)
# timing / checkpoints
parser.add_argument('--report', type=int, nargs='+', default=[10, 60, 300, 600],
                    help='Checkpoints (seconds) where the model is evaluated / saved')
parser.add_argument('--num_eval_per_cp', type=int, default=5,
                    help='Number of evaluations per checkpoint')
# optimisation
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau',
                    choices=['StepLR', 'ReduceLROnPlateau', 'OneCycleLR', 'None'])
# SG / LD specifics
parser.add_argument('--method', type=str, default='SPOPlus', choices=['SPOPlus', 'IMLE', 'Exact'])
parser.add_argument('--n_samples', type=int, default=1)     # IMLE
parser.add_argument('--lambda_imle', type=float, default=10)
parser.add_argument('--sigma', type=float, default=1.0)
parser.add_argument('--step_mu', type=int, default=10)      # SG
parser.add_argument('--n_iter_mu', type=int, default=30)
parser.add_argument('--muloss', type=int, default=1)
# outputs
parser.add_argument('--out_file', default='portfolio/results_new.csv')
parser.add_argument('--wandb', type=int, default=0, help='1 to enable wandb (offline)')
args = parser.parse_args()

# UTILS
seed_everything(args.seed)
muloss_bool = bool(args.muloss)

# datasets parameters
num_feat          = 5
num_item          = args.n
num_data_train    = 100
num_data_eval     = 25
num_data_test     = 10000
gamma             = 2.25
gamma_str         = str(gamma).replace('.', '-')

def ensure_datasets_exist():
    if args.regenerate:
        from portfolio.gen_data import gen_datafile
        os.makedirs("portfolio/datasets", exist_ok=True)
        print("[regen] generating datasets …")
        gen_datafile(num_data_train, num_data_eval, num_data_test,
                     num_feat, num_item, args.deg, gamma, 1000,
                     principal_lin=0, verbose=False)

def load_sets():
    """Returns train_loader, eval_loader, test_loader, and the covariance matrix."""
    train_set = ImportDataset(f"portfolio/datasets/train_{num_item}_{num_data_train}_{num_feat}_{args.deg}_{gamma_str}.txt")
    eval_set  = ImportDataset(f"portfolio/datasets/validation_{num_item}_{num_data_eval}_{num_feat}_{args.deg}_{gamma_str}.txt")
    test_set  = ImportDataset(f"portfolio/datasets/test_{num_item}_{num_data_test}_{num_feat}_{args.deg}_{gamma_str}.txt")
    return (
        train_set.get_dataloader(batch_size=32, shuffle=True),
        eval_set.get_dataloader(batch_size=32, shuffle=False),
        test_set.get_dataloader(batch_size=32, shuffle=False),
        1e5 * train_set.get_cov()
    )

def make_scheduler(approach,opt):
    if args.scheduler == "ReduceLROnPlateau":
        patience = 100 if approach != "classic" else 10
        return optim.lr_scheduler.ReduceLROnPlateau(opt, patience=patience, factor=0.5, min_lr=1e-6)
    elif args.scheduler == "StepLR":
        return optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
    elif args.scheduler == "OneCycleLR":
        return optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.ep_classic or 1)
    return None

def build_diff_method(name, approach, num_item, cov, gamma, principal_lin=False):
    """Returns an instance of I_MLE / SPOPlus / Exact."""
    if approach == "LD" or approach == "SG":
        if name == "IMLE":
            solver = Solveur_lin(cov.shape[0], maximize=True) if principal_lin else Solveur_quad(cov.shape[0], cov, gamma)
            return I_MLE(solver, device,
                        n_samples=args.n_samples, lambd=args.lambda_imle, sigma=args.sigma)
        if name == "SPOPlus":
            solver = Solveur_lin(cov.shape[0], maximize=False) if principal_lin else Solveur_quad(cov.shape[0], cov, gamma)
            return SPOPlus(solver, device)
        if name == "Exact":
            solver = BatchSolverExact(num_item, cov, gamma, device)
            return Exact(solver, device)
        raise ValueError("unknown method")
    elif approach == "classic":
        if name == "IMLE":
            solver = portfolioModel(num_assets=cov.shape[0], covariance=cov, gamma=gamma) 
            return I_MLE(solver, device,
                        n_samples=args.n_samples, lambd=args.lambda_imle, sigma=args.sigma)
        elif name == "SPOPlus":
            solver = portfolioModel(num_assets=cov.shape[0], covariance=cov, gamma=gamma) 
            return SPOPlus(solver, device)
    elif approach == "MSE":
        return MSE()

def wandb_init(job_name, cfg):
    if not args.wandb:
        return None
    import wandb
    return wandb.init(
        mode="offline",
        project="DFL_portfolio_newTrain",
        name=job_name,
        config=cfg,
        dir="./"
    )

# run train
def run_train(approach:str, num_epochs:int):
    """
    approach ∈ {'classic','LD','SG','MSE'} – num_epochs is kept even if train()
    stops early due to checkpoints (useful for OneCycleLR).
    """
    if num_epochs <= 0:
        return

    print(f"\n=== {approach.upper()} experiment – {num_epochs} epochs ===")
    # data & solver
    train_loader, eval_loader, test_loader, cov = load_sets()
    eval_solver = portfolioModel(num_assets=num_item, covariance=cov, gamma=gamma)

    # model, opti, sched
    model = CustomMLP([num_feat, num_item]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(approach,optimizer)

    # diff_list
    diff = build_diff_method(args.method, approach, num_item, cov, gamma, principal_lin=False)
    diff_list = [diff]

    # SG specifics
    opt_mu = None
    mu0    = None
    if approach == "SG":
        lin_solver  = BatchSolverLin(num_item, device)
        quad_solver = BatchSolverQuad(num_item, cov, gamma, device)
        exact_solver = BatchSolverExact(num_item, cov, gamma, device)
        solvers = [quad_solver, lin_solver] if args.method != "Exact" else [exact_solver, lin_solver]
        opt_mu  = OptimizationBatchModel_serial(solvers)
        mu0     = torch.ones(len(train_loader.dataset), len(solvers), cov.shape[0]-1, num_item,
                             device=device, dtype=torch.float32)

    # wandb
    run = wandb_init(
        f"{args.method}_{approach}_{num_item}_{num_data_train}_{num_feat}_{args.deg}_{gamma_str}",
        dict(
            approach    = approach,
            diff_method = args.method ,
            lr          = args.lr,
            n           = num_item,
            deg         = args.deg,
            checkpoints = args.report
        )
    )

    # CSV parameters
    param_csv = dict(
        n        = num_item,
        jobtype  = approach,
        method   = args.method if approach != "MSE" else "",
        lr       = args.lr,
        muloss   = muloss_bool,
        step_mu  = args.step_mu if approach == "SG" else 0,
        num_iter_mu = args.n_iter_mu if approach == "SG" else 0,
    )

    # TRAIN call
    train(
        model, diff_list, eval_solver,
        train_loader, eval_loader, test_loader,
        optimizer, scheduler,
        checkpoints=args.report,
        num_eval_per_cp=args.num_eval_per_cp,
        output_file=args.out_file,
        approach=approach,
        loss_with_mu=muloss_bool,
        decompositions=[0],                # only one decomposition here
        freq_dec_change=1,
        step_mu=args.step_mu,
        num_iter_mu=args.n_iter_mu,
        optimizer_mu=opt_mu,
        mu_global0=mu0,
        run=run, verbose=True,
        param=param_csv, metric="time",
        device=device
    )

    if run is not None:
        run.finish()

# run experiments
ensure_datasets_exist()

run_train("classic", args.ep_classic)
run_train("LD",      args.ep_ld)
run_train("SG",      args.ep_sg)
run_train("MSE",     args.ep_mse)

print("\n✔ All requested trainings are finished.")
