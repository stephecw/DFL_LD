import argparse, os, csv, time, torch, numpy as np
from torch import optim
from pyepo.model.grb import knapsackModel
from pyepo.func.utlis import sumGammaDistribution

from knapsack.data_import import ImportDataset
from train import train, test
from models_class import CustomMLP
from diff_methods import I_MLE, SPOPlus, MSE
from opti_X_mu_CPU import OptimizationBatchModel_serial
from knapsack.solver import solver_X_knapsack
from utils import seed_everything

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Training on:", device)

# ARGPARSE
parser = argparse.ArgumentParser("Knapsack experiments with unified train()")
# data dimensions
parser.add_argument('--dim', type=int, default=10, help='Number of constraints.')
parser.add_argument('--n', type=int, default=50, help='Number of items.')
parser.add_argument('--n_feat', type=int, default=12, help='Number of features.')
parser.add_argument('--n_train', type=int, default=200, help='Number of training instances.')
parser.add_argument('--n_eval', type=int, default=100, help='Number of eval instances.')
parser.add_argument('--n_test', type=int, default=1000, help='Number of test instances.')
parser.add_argument('--deg', type=int, default=4, help='Polynomial degree.')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--regenerate', type=int, default=0,
                    help='1 to (re)generate datasets via knapsack.gen_data before training')
parser.add_argument('--n_iter_gen', type=int, default=200,
                    help='Number of mu-optimization iterations for data generation (used if --regenerate 1)')
# decomposition
parser.add_argument('--keep', type=int, default=1,
                    help='Number of constraints in the main subproblem. Use -1 for the multi-decomposition file.')
parser.add_argument('--mains', type=int, nargs='+', default=[0],
                    help='Indices of decompositions to use during training. '
                         'For --keep K (>=1): single decomposition file train_{dim}_{K}_{mains[0]}_{...}.txt. '
                         'For --keep -1: subset of indices in [0, dim-1] picked from train_{dim}_-1_{...}.txt.')
parser.add_argument('--combine', type=str, default='random',
                    help='How to combine decompositions across epochs (only "random" supported; '
                         'mapped to freq_dec_change=1 in train()).')
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
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau',
                    choices=['StepLR', 'ReduceLROnPlateau', 'OneCycleLR', 'None'])
# diff method
parser.add_argument('--diff', type=str, default='SPOPlus', choices=['SPOPlus', 'IMLE'])
parser.add_argument('--n_samples', type=int, default=1)            # IMLE
parser.add_argument('--lambd', type=float, default=10)             # IMLE
parser.add_argument('--sigma', type=float, default=1.0)            # IMLE
parser.add_argument('--kappa', type=int, default=5)                # IMLE noise distribution
# SG specifics
parser.add_argument('--step_mu', type=int, default=10)
parser.add_argument('--n_iter_mu', type=int, default=30)
parser.add_argument('--muloss', type=int, default=1, help='If 1, include mu_sum in the loss')
# outputs
parser.add_argument('--out_file', default='knapsack/results.csv')
parser.add_argument('--save_model', type=int, default=1, help='1 to save the best model state_dict after train()')
parser.add_argument('--wandb', type=int, default=0, help='1 to enable wandb (offline)')
args = parser.parse_args()

seed_everything(args.seed)
muloss_bool = bool(args.muloss)

num_feat       = args.n_feat
num_item       = args.n
dim            = args.dim
deg            = args.deg
keep           = args.keep
mains          = args.mains
num_data_train = args.n_train
num_data_eval  = args.n_eval
num_data_test  = args.n_test

# IMLE extra args
diff_method_extra = {}
if args.diff == "IMLE":
    diff_method_extra = dict(
        n_samples=args.n_samples,
        sigma=args.sigma,
        lambd=args.lambd,
        distribution=sumGammaDistribution(args.kappa),
    )


def ensure_datasets_exist():
    """Generate base + train + eval + test datasets if --regenerate 1."""
    if not args.regenerate:
        return
    from knapsack.gen_data import gen_base_data, add_X_mu_single, add_X_mu_multiple
    os.makedirs("knapsack/datasets", exist_ok=True)
    print("[regen] generating base datasets …")
    gen_base_data(num_data_train, num_data_eval, num_data_test,
                  num_feat, num_item, dim, deg=deg, noise_width=0.5, verbose=True)
    if keep == -1:
        print(f"[regen] adding X, mu (multi-decomposition) …")
        add_X_mu_multiple(num_data_train, num_feat, num_item, dim, deg,
                          num_iter=args.n_iter_gen, convergence=1e-4,
                          monitor=False, verbose=True)
    else:
        for m in mains:
            print(f"[regen] adding X, mu (keep={keep}, main={m}) …")
            add_X_mu_single(num_data_train, num_feat, num_item, dim, deg,
                            keep=keep, main=m,
                            num_iter=args.n_iter_gen, convergence=1e-4,
                            monitor=False, verbose=True)


def load_sets():
    """Returns train_loader, eval_loader, test_loader, weights, capacities."""
    if keep == -1:
        train_fname = f"knapsack/datasets/train_{dim}_-1_{num_feat}_{num_item}_{num_data_train}_{deg}.txt"
    else:
        if len(mains) != 1:
            raise ValueError(
                f"For --keep {keep} (>=1) only a single main is supported (got --mains {mains}). "
                "Use --keep -1 for multi-decomposition."
            )
        train_fname = f"knapsack/datasets/train_{dim}_{keep}_{mains[0]}_{num_feat}_{num_item}_{num_data_train}_{deg}.txt"
    eval_fname = f"knapsack/datasets/eval_{dim}_{num_feat}_{num_item}_{num_data_eval}_{deg}.txt"
    test_fname = f"knapsack/datasets/test_{dim}_{num_feat}_{num_item}_{num_data_test}_{deg}.txt"

    print(f"Loading {train_fname}", flush=True)
    train_set = ImportDataset(train_fname, device)
    print(f"Loading {eval_fname}", flush=True)
    eval_set  = ImportDataset(eval_fname,  device)
    print(f"Loading {test_fname}", flush=True)
    test_set  = ImportDataset(test_fname,  device)

    weights    = train_set.get_weights(tensor=True)
    capacities = train_set.get_capacities(tensor=True)
    return (
        train_set.get_dataloader(batch_size=32, shuffle=True),
        eval_set.get_dataloader(batch_size=32, shuffle=False),
        test_set.get_dataloader(batch_size=32, shuffle=False),
        weights, capacities,
    )


def make_scheduler(approach, opt):
    if args.scheduler == "ReduceLROnPlateau":
        patience = 100 if approach != "classic" else 10
        return optim.lr_scheduler.ReduceLROnPlateau(opt, patience=patience, factor=0.5, min_lr=1e-6)
    elif args.scheduler == "StepLR":
        return optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
    elif args.scheduler == "OneCycleLR":
        return optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=1000)
    return None


def build_diff_methods(approach, weights, capacities):
    """Build the list of differentiation methods, one per decomposition position.

    For approach == 'classic': single diff method on the full problem.
    For approach == 'LD'/'SG' with keep != -1: single diff method on the main subproblem (first `keep` constraints).
    For approach == 'LD'/'SG' with keep == -1: one diff method per constraint (each main has 1 constraint).
    For approach == 'MSE': just MSE().
    """
    if approach == "MSE":
        return [MSE()]

    if approach == "classic":
        solver = knapsackModel(weights=weights, capacity=capacities)
        if args.diff == "IMLE":
            return [I_MLE(solver, device, **diff_method_extra)]
        return [SPOPlus(solver, device)]

    # LD / SG
    if keep == -1:
        # one solver per constraint (each decomposition's main = 1 constraint)
        diff_list = [None] * dim
        for i in range(dim):
            sub_solver = knapsackModel(weights=weights[i:i+1], capacity=capacities[i:i+1])
            if args.diff == "IMLE":
                diff_list[i] = I_MLE(sub_solver, device, **diff_method_extra)
            else:
                diff_list[i] = SPOPlus(sub_solver, device)
        return diff_list
    else:
        # single decomposition: only one diff method, for decomp index 0
        sub_solver = knapsackModel(weights=weights[:keep], capacity=capacities[:keep])
        if args.diff == "IMLE":
            return [I_MLE(sub_solver, device, **diff_method_extra)]
        return [SPOPlus(sub_solver, device)]


def wandb_init(job_name, cfg):
    if not args.wandb:
        return None
    import wandb
    return wandb.init(
        mode="offline",
        project="DFL_knapsack_newTrain",
        name=job_name,
        config=cfg,
        dir="./",
    )


def run_train(approach, num_epochs):
    """approach in {'classic','LD','SG','MSE'}; num_epochs is informational only."""
    if num_epochs <= 0:
        return

    print(f"\n=== {approach.upper()} experiment – {num_epochs} epochs ===", flush=True)

    train_loader, eval_loader, test_loader, weights, capacities = load_sets()
    eval_solver = knapsackModel(weights=weights, capacity=capacities)

    model = CustomMLP([num_feat, num_item]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = make_scheduler(approach, optimizer)

    diff_list = build_diff_methods(approach, weights, capacities)

    # decompositions to use during training
    if approach in ("classic", "MSE"):
        decompositions = [0]
    elif keep == -1:
        decompositions = list(mains)  # subset of [0, ..., dim-1]
    else:
        decompositions = [0]

    # SG specifics
    opt_mu = None
    mu0 = None
    if approach == "SG":
        # one solver per constraint, used by the inner mu-optimizer
        sub_solvers = [
            solver_X_knapsack(np.expand_dims(weights[i].cpu().numpy(), axis=0),
                              np.expand_dims(capacities[i].cpu().numpy(), axis=0))
            for i in range(dim)
        ]
        opt_mu = OptimizationBatchModel_serial(sub_solvers)
        # mu_global has full dim slots so it can be indexed by absolute decomposition index
        mu0 = torch.ones(len(train_loader.dataset), dim, dim - 1, num_item,
                         device=device, dtype=torch.float32)

    keep_str = str(keep) if keep != -1 else "multi"
    job_name = f"{args.diff}_{approach}_{dim}_{keep_str}_{num_feat}_{num_item}_{num_data_train}_{deg}"
    run = wandb_init(
        job_name,
        dict(
            approach    = approach,
            diff_method = args.diff,
            lr          = args.lr,
            n           = num_item,
            dim         = dim,
            keep        = keep,
            mains       = mains,
            deg         = deg,
            checkpoints = args.report,
        ),
    )

    param_csv = dict(
        n           = num_item,
        dim         = dim,
        keep        = keep,
        mains       = "_".join(map(str, mains)),
        deg         = deg,
        jobtype     = approach,
        method      = args.diff if approach != "MSE" else "",
        lr          = args.lr,
        muloss      = muloss_bool,
        step_mu     = args.step_mu if approach == "SG" else 0,
        num_iter_mu = args.n_iter_mu if approach == "SG" else 0,
    )

    result = train(
        model, diff_list, eval_solver,
        train_loader, eval_loader, test_loader,
        optimizer, scheduler,
        checkpoints=args.report,
        num_eval_per_cp=args.num_eval_per_cp,
        output_file=args.out_file,
        approach=approach,
        loss_with_mu=muloss_bool,
        decompositions=decompositions,
        freq_dec_change=1,
        step_mu=args.step_mu,
        num_iter_mu=args.n_iter_mu,
        optimizer_mu=opt_mu,
        mu_global0=mu0,
        run=run, verbose=True,
        param=param_csv, metric="time",
        device=device,
    )

    if args.save_model and result and isinstance(result, dict) and result.get('best_model_state') is not None:
        os.makedirs("knapsack/models", exist_ok=True)
        model.load_state_dict(result['best_model_state'])
        method_tag = "MSE" if approach == "MSE" else f"{args.diff}_{approach}"
        fname = f"knapsack/models/{method_tag}_{dim}_{keep_str}_{num_feat}_{num_item}_{num_data_train}_{deg}.pth"
        torch.save(model.state_dict(), fname)
        print(f"Saved best model to {fname}", flush=True)

    if run is not None:
        run.finish()


# run experiments
ensure_datasets_exist()

run_train("classic", args.ep_classic)
run_train("LD",      args.ep_ld)
run_train("SG",      args.ep_sg)
run_train("MSE",     args.ep_mse)

print("\n✔ All requested trainings are finished.")
