import wandb
import torch
from torch import optim
from torch.utils.data import DataLoader
from data_import import ImportDataset
from train_imle import train, CustomMLP

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sweep configuration
def get_sweep_config():
    return {
        'method': 'bayes',
        'metric': {
            'name': 'regret',
            'goal': 'minimize'
        },
        'parameters': {
            # Hyperparameters to tune
            'lr': {
                'distribution': 'log_uniform',
                'min': 1e-5,
                'max': 1e-1
            },
            'IMLE_n_samples': {
                'distribution': 'int',
                'min': 1,
                'max': 50
            },
            'IMLE_sigma': {
                'distribution': 'log_uniform',
                'min': 0.01,
                'max': 10
            },
            'IMLE_lambd': {
                'distribution': 'log_uniform',
                'min': 0.1,
                'max': 100
            },
            # Static problem parameters
            'dim': {'value': 5},
            'num_feat': {'value': 200},
            'num_item': {'value': 30},
            'num_data_train': {'value': 500},
            'num_data_test': {'value': 100},
            'hidden_layer': {'value': 100},
            'batch_size': {'value': 32},
            'epochs': {'value': 20},
            'sched_step_size': {'value': 10},
            'sched_gamma': {'value': 0.1},
            'IMLE_two_sides': {'value': False},
            'IMLE_processes': {'value': 1}
        }
    }

# Training function for the sweep
def train_model():
    # Initialize a new W&B run
    with wandb.init() as run:
        config = run.config

        # Load datasets
        train_set = ImportDataset(
            f"datasets/train_{config.dim}_{config.num_feat}_{config.num_item}_{config.num_data_train}.txt"
        )
        test_set = ImportDataset(
            f"datasets/test_{config.dim}_{config.num_feat}_{config.num_item}_{config.num_data_test}.txt"
        )
        train_loader = train_set.get_dataloader(batch_size=config.batch_size, shuffle=True)
        test_loader = test_set.get_dataloader(batch_size=config.batch_size, shuffle=False)

        # Problem parameters
        weights = train_set.get_weights(tensor=True)
        capacities = train_set.get_capacities(tensor=True)

        # Model
        model = CustomMLP([config.num_feat, config.hidden_layer, config.num_item]).to(device)

        # Optimizer & scheduler
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.sched_step_size,
            gamma=config.sched_gamma
        )

        # Train with i-MLE
        train(
            model, run,
            train_loader, test_loader,
            optimizer, scheduler,
            weights, capacities,
            epochs=config.epochs,
            IMLE_n_samples=config.IMLE_n_samples,
            IMLE_sigma=config.IMLE_sigma,
            IMLE_lambd=config.IMLE_lambd,
            IMLE_two_sides=config.IMLE_two_sides,
            IMLE_processes=config.IMLE_processes,
            verbose=False
        )

# Entry point: launch sweep
if __name__ == '__main__':
    sweep_config = get_sweep_config()
    sweep_id = wandb.sweep(
        sweep_config,
        project='DFL_hyperopt',
        entity='hugoper-polytechnique-montr-al'
    )
    wandb.agent(sweep_id, function=train_model, count=20)