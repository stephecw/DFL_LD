from math import log
import optuna
import torch
from torch import optim
from torch.utils.data import DataLoader
from data_import import ImportDataset
from train_imle import train, train_LD, test_regret, CustomMLP

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("→ Entraînement sur :", device)

# Static problem parameters
DIM = 5
NUM_FEAT = 200
NUM_ITEM = 50
NUM_DATA_TRAIN = 500
NUM_DATA_TEST = 100
HIDDEN_LAYER = 100
BATCH_SIZE = 32
EPOCHS_LD = 50
EPOCHS_classic = 3
SCHED_STEP_SIZE = 10
SCHED_GAMMA = 0.1
IMLE_TWO_SIDES = False
IMLE_PROCESSES = 1

# Objective for classic i-MLE
def objective_classic(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    n_samples = trial.suggest_int('IMLE_n_samples', 1, 20)
    sigma = trial.suggest_float('IMLE_sigma', 0.01, 10, log=True)
    lambd = trial.suggest_float('IMLE_lambd', 0.1, 100, log=True)

    # Load data
    train_set = ImportDataset(f"datasets/train_{DIM}_{NUM_FEAT}_{NUM_ITEM}_{NUM_DATA_TRAIN}.txt")
    test_set = ImportDataset(f"datasets/test_{DIM}_{NUM_FEAT}_{NUM_ITEM}_{NUM_DATA_TEST}.txt")
    train_loader = train_set.get_dataloader(batch_size=BATCH_SIZE, shuffle=True)
    test_loader = test_set.get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    # Problem parameters
    weights = train_set.get_weights(tensor=True)
    capacities = train_set.get_capacities(tensor=True)

    # Model
    model = CustomMLP([NUM_FEAT, HIDDEN_LAYER, NUM_ITEM]).to(device)

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = None #optim.lr_scheduler.StepLR(optimizer, step_size=SCHED_STEP_SIZE, gamma=SCHED_GAMMA)

    # Train
    train(
        model, None,
        train_loader, test_loader,
        optimizer, scheduler,
        weights, capacities,
        epochs=EPOCHS_classic,
        IMLE_n_samples=n_samples,
        IMLE_sigma=sigma,
        IMLE_lambd=lambd,
        IMLE_two_sides=IMLE_TWO_SIDES,
        IMLE_processes=IMLE_PROCESSES,
        verbose=False
    )

    # Evaluate regret
    regret = test_regret(model, test_loader, weights, capacities)
    return regret

# Objective for LD
def objective_LD(trial):
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    n_samples = trial.suggest_int('IMLE_n_samples', 1, 20)
    sigma = trial.suggest_float('IMLE_sigma', 0.01, 10, log=True)
    lambd = trial.suggest_float('IMLE_lambd', 0.1, 100, log=True)

    # Load data
    train_set = ImportDataset(f"datasets/train_{DIM}_{NUM_FEAT}_{NUM_ITEM}_{NUM_DATA_TRAIN}.txt")
    test_set = ImportDataset(f"datasets/test_{DIM}_{NUM_FEAT}_{NUM_ITEM}_{NUM_DATA_TEST}.txt")
    train_loader = train_set.get_dataloader(batch_size=BATCH_SIZE, shuffle=True)
    test_loader = test_set.get_dataloader(batch_size=BATCH_SIZE, shuffle=False)

    # Problem parameters
    weights = train_set.get_weights(tensor=True)
    capacities = train_set.get_capacities(tensor=True)

    # Model
    model = CustomMLP([NUM_FEAT, HIDDEN_LAYER, NUM_ITEM]).to(device)

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = None # optim.lr_scheduler.StepLR(optimizer, step_size=SCHED_STEP_SIZE, gamma=SCHED_GAMMA)

    # Train with LD
    train_LD(
        model, None,
        train_loader, test_loader,
        optimizer, scheduler,
        weights, capacities,
        epochs=EPOCHS_LD,
        IMLE_n_samples=n_samples,
        IMLE_sigma=sigma,
        IMLE_lambd=lambd,
        IMLE_two_sides=IMLE_TWO_SIDES,
        IMLE_processes=IMLE_PROCESSES,
        verbose=False
    )

    # Evaluate regret
    regret = test_regret(model, test_loader, weights, capacities)
    return regret


def main():
        # Study for LD
    study_ld = optuna.create_study(direction='minimize', study_name='LD_iMLE')
    study_ld.optimize(objective_LD, n_trials=20) # 20
    best_ld = study_ld.best_params
    
    # Save to CSV
    import csv
    with open('best_hyperparams_LD.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['study', 'lr', 'IMLE_n_samples', 'IMLE_sigma', 'IMLE_lambd','dim', 'num_feat', 'num_item', 'num_data_train'])
        writer.writerow([
            'LD',
            best_ld['lr'],
            best_ld['IMLE_n_samples'],
            best_ld['IMLE_sigma'],
            best_ld['IMLE_lambd'],
            DIM, NUM_FEAT, NUM_ITEM, NUM_DATA_TRAIN
        ])
    
    print("Best hyperparameters saved to best_hyperparams_LD.csv")


    # Study for classic i-MLE
    study_classic = optuna.create_study(direction='minimize', study_name='classic_iMLE')
    study_classic.optimize(objective_classic, n_trials=20) #20
    best_classic = study_classic.best_params

    # Save to CSV
    with open('best_hyperparams_classic.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['study', 'lr', 'IMLE_n_samples', 'IMLE_sigma', 'IMLE_lambd','dim', 'num_feat', 'num_item', 'num_data_train'])
        writer.writerow([
            'classic',
            best_classic['lr'],
            best_classic['IMLE_n_samples'],
            best_classic['IMLE_sigma'],
            best_classic['IMLE_lambd'],
            DIM, NUM_FEAT, NUM_ITEM, NUM_DATA_TRAIN
        ])
    print("Best hyperparameters saved to best_hyperparams_classic.csv")
    

if __name__ == '__main__':
    main()