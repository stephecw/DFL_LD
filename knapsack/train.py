import torch
from torch import nn
from pyepo.model.grb import knapsackModel
import gurobipy as gp
from pyepo.func import implicitMLE, SPOPlus
from opti_X_mu import OptimizationBatchModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modèle prédictif (régression linéaire)
class LinearRegression(nn.Module):
    def __init__(self, num_feat, num_item):
        super().__init__()
        self.linear = nn.Linear(num_feat, num_item)

    def forward(self, x):
        return self.linear(x)
    
class CustomMLP(nn.Module):
    def __init__(self, layer_sizes, activation=nn.ReLU, dropout=0.0):
        super(CustomMLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # Ajouter une activation sauf pour la dernière couche
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, run, dataloader_train, dataloader_test, optimizer, scheduler, weights, capacities, 
          epochs, time_limit,
          diff_method, diff_method_arg,
          verbose=False):
    """
        Entraînement du modèle avec i-MLE classique
        model: modèle prédictif des profits
        run: wandb.run pour l'enregistrement des résultats
        dataloader: DataLoader avec (z, c, x, X1*(c), mu(c))
        optimizer: optimiseur PyTorch
        scheduler: planificateur d'apprentissage
        weights: matrice [m, n] des poids
        capacities: vecteur [m] des capacités
        epochs: nombre d’époques d'entraînement
        IMLE_n_samples: nombre d'échantillons pour i-MLE
        IMLE_sigma: sigma pour i-MLE
        IMLE_lambd: lambda pour i-MLE
        IMLE_two_sides: bool : Si True, utilise une perturbation à deux côtés
        IMLE_processes: int : Nombre de processus pour i-MLE
        verbose: bool : Si True, affiche les détails de l'entraînement
    """
    time_monitoring = run is not None or time_limit is not None
    
    # multiKPModel prend en charge plusieurs contraintes
    optmodel = knapsackModel(weights=weights, capacity=capacities)
    # i-MLE avec solveur exact multi-contrainte
    if diff_method == "IMLE":
        diff = implicitMLE(optmodel, *diff_method_arg)
    elif diff_method == "SPOPlus":
        diff = SPOPlus(optmodel, *diff_method_arg)

    if time_monitoring:
        import time
        start_time = time.time()
        train_time = 0
        
    for epoch in range(epochs):
        if time_monitoring:
            epoch_start_time = time.time()
            
        model.train()
        total_loss = 0
        total_grad_norm = 0.0
        
        for z, c, x, X1, mu in dataloader_train:
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            cp = model(z)
            xp = diff(cp).to(device)

            # Regret = c · (w - wp)
            loss = torch.sum(c * (x - xp), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2

            total_loss += loss.item()
            
        if scheduler is not None:
            scheduler.step()
        
        if time_monitoring:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            train_time += epoch_duration
        
        mean_loss = total_loss / len(dataloader_train)
        if time_monitoring:
            total_grad_norm = total_grad_norm ** 0.5
            current_lr = get_learning_rate(optimizer)            
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration, "train_time": train_time, "grad_norm": total_grad_norm, "lr": current_lr})

        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")
            
        with torch.no_grad():
            model.eval()
            total_regret = 0
            total_count = 0
            relat_regrets = []
            for z, c, x, _, _ in dataloader_test:
                z, c, x = [t.to(device) for t in (z, c, x)]
                c_hat = model(z)  # prédiction des coûts [batch, n]
                
                for i in range(z.size(0)):
                    model_i = knapsackModel(weights=weights, capacity=capacities)
                    c_numpy = c_hat[i].detach().cpu().numpy()
                    model_i.setObj(c_numpy)
                    x_hat_np, _ = model_i.solve()

                    x_true = x[i].to(dtype=torch.float32, device=device)
                    c_true = c[i].to(dtype=torch.float32, device=device)

                    x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32,
                                            device=device)

                    relat_regret = torch.dot(c_true, x_true - x_hat_tensor)/torch.dot(c_true, x_true)
                    relat_regrets.append(relat_regret)

            relat_regrets = torch.stack(relat_regrets)
                
        mean_relat_regret = relat_regrets.mean().item()
        std_relat_regret = relat_regrets.std().item()
        if run is not None:
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "Mean relative regret": mean_relat_regret, "Std relative regret": std_relat_regret, "train_time": train_time})
        if verbose:
            print(f"Eval Epoch {epoch} | Regret relatif moyen : {mean_relat_regret,:.4f}")
            
        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Limite de temps atteinte, arrêt de l'entraînement.")
                return
            
    if run is not None:   
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})


def train_LD(model, run, dataloader_train, dataloader_test, optimizer, scheduler, weights, capacities, 
          epochs, time_limit,
          diff_method, diff_method_arg,
          verbose=False):
    """
        Entraînement du modèle avec la décomposition lagrangienne
        model: modèle prédictif des profits
        run: wandb.run pour l'enregistrement des résultats
        dataloader: DataLoader avec (z, c, x, X1*(c), mu(c))
        optimizer: optimiseur PyTorch
        scheduler: planificateur d'apprentissage
        weights: matrice [m, n] des poids
        capacities: vecteur [m] des capacités
        epochs: nombre d’époques d'entraînement
        IMLE_n_samples: nombre d'échantillons pour i-MLE
        IMLE_sigma: sigma pour i-MLE
        IMLE_lambd: lambda pour i-MLE
        IMLE_two_sides: bool : Si True, utilise une perturbation à deux côtés
        IMLE_processes: int : Nombre de processus pour i-MLE
        verbose: bool : Si True, affiche les détails de l'entraînement
    """
    time_monitoring = run is not None or time_limit is not None
    
    # Créer un solveur i-MLE avec les mu du batch
    solver = knapsackModel(weights[0].unsqueeze(0), capacities[0].unsqueeze(0))
    if diff_method == "IMLE":
        diff = implicitMLE(solver, *diff_method_arg)
    elif diff_method == "SPOPlus":
        diff = SPOPlus(solver, *diff_method_arg)

    if time_monitoring:
        import time
        start_time = time.time()
        train_time = 0
    
    for epoch in range(epochs):
        if run is not None:
            epoch_start_time = time.time()
        model.train()
        total_loss = 0
        total_grad_norm = 0.0

        for z, c, x, X1, mu in dataloader_train: # x vrai solution et X1 solution avec mu(c)
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            c_hat = model(z)  # prédiction des profits ĉ


            # Résolution avec i-MLE
            mu_sum = mu.sum(dim=1)# shape [batch, n]
            X1p = diff(c_hat + mu_sum).to(device)   # x̂ obtenu avec solve_main_problem

            # (c + sum mu_i for i ≥ 2) · (w - x̂)
            profit_modified = c + mu_sum     # shape [batch, n]
            loss = torch.sum(profit_modified * (X1 - X1p), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()
        
        if time_monitoring:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            train_time += epoch_duration
        
        mean_loss = total_loss / len(dataloader_train)
        if time_monitoring:
            total_grad_norm = total_grad_norm ** 0.5
            current_lr = get_learning_rate(optimizer)            
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration, "train_time": train_time, "grad_norm": total_grad_norm, "lr": current_lr})

        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")
        with torch.no_grad():
            model.eval()
            total_regret = 0
            total_count = 0
            relat_regrets = []
            for z, c, x, _, _ in dataloader_test:
                z, c, x = [t.to(device) for t in (z, c, x)]
                c_hat = model(z)  # prédiction des coûts [batch, n]
                
                for i in range(z.size(0)):
                    model_i = knapsackModel(weights=weights, capacity=capacities)
                    c_numpy = c_hat[i].detach().cpu().numpy()
                    model_i.setObj(c_numpy)
                    x_hat_np, _ = model_i.solve()

                    x_true = x[i].to(dtype=torch.float32, device=device)
                    c_true = c[i].to(dtype=torch.float32, device=device)

                    x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32,
                                            device=device)

                    relat_regret = torch.dot(c_true, x_true - x_hat_tensor)/torch.dot(c_true, x_true)
                    relat_regrets.append(relat_regret)

            relat_regrets = torch.stack(relat_regrets)
                
        mean_relat_regret = relat_regrets.mean().item()
        std_relat_regret = relat_regrets.std().item()
        if run is not None:
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "Mean relative regret": mean_relat_regret, "Std relative regret": std_relat_regret, "train_time": train_time})
        if verbose:
            print(f"Eval Epoch {epoch} | Regret relatif moyen : {mean_relat_regret,:.4f}")
            
        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Limite de temps atteinte, arrêt de l'entraînement.")
                return
            
    if run is not None:   
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})


def train_SG(model, run, dataloader_train, dataloader_test, optimizer, scheduler, weights, capacities, 
          epochs, time_limit,
          diff_method, diff_method_arg,
          step_mu=10, n_iter_mu = 100,
          verbose=False):
    """
        Entraînement du modèle avec la décomposition lagrangienne
        model: modèle prédictif des profits
        run: wandb.run pour l'enregistrement des résultats
        dataloader: DataLoader avec (z, c, x, X1*(c), mu(c))
        optimizer: optimiseur PyTorch
        scheduler: planificateur d'apprentissage
        weights: matrice [m, n] des poids
        capacities: vecteur [m] des capacités
        epochs: nombre d’époques d'entraînement
        IMLE_n_samples: nombre d'échantillons pour i-MLE
        IMLE_sigma: sigma pour i-MLE
        IMLE_lambd: lambda pour i-MLE
        IMLE_two_sides: bool : Si True, utilise une perturbation à deux côtés
        IMLE_processes: int : Nombre de processus pour i-MLE
        verbose: bool : Si True, affiche les détails de l'entraînement
    """
    time_monitoring = run is not None or time_limit is not None
    
    # Créer un solveur i-MLE avec les mu du batch
    solver = knapsackModel(weights[0].unsqueeze(0), capacities[0].unsqueeze(0))
    if diff_method == "IMLE":
        diff = implicitMLE(solver, *diff_method_arg)
    elif diff_method == "SPOPlus":
        diff = SPOPlus(solver, *diff_method_arg)
        
    dim = weights.shape[0]
    n_items = weights.shape[1]

    mu_global = torch.ones(len(dataloader_train.dataset), dim - 1, n_items, device=device, dtype = torch.float32)
    
    if time_monitoring:
        import time
        start_time = time.time()
        train_time = 0

    for epoch in range(epochs):
        if run is not None:
            epoch_start_time = time.time()
        model.train()
        total_loss = 0
        total_grad_norm = 0.0

        for batch_idx, (z, c, x, X1, mu) in enumerate(dataloader_train): # x vrai solution et X1 solution avec mu(c)
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            c_hat = model(z)  # prédiction des profits ĉ

            idx = (batch_idx * dataloader_train.batch_size + torch.arange(z.size(0), device=device))
            mu_tilde = mu_global[idx]

            # Mise à jour de mu_global
            if epoch % step_mu == 0:
                optimizer_mu = OptimizationBatchModel(
                    mu_init=mu_tilde.clone(),
                    num_item=n_items,
                    dim=dim,
                    c_batch=c_hat.detach(),
                    weights=weights,
                    capacities=capacities
                )
                optimizer_mu.optim_mu(verbose=False, max_iter=n_iter_mu)

                mu_tilde = optimizer_mu.get_mu().detach()
                mu_global[idx] = mu_tilde


            # Résolution avec i-MLE
            X1p = diff(c_hat + mu_tilde.sum(dim=1)).to(device)   # x̂ obtenu avec solve_main_problem

            # (c + sum mu_i for i ≥ 2) · (w - x̂)
            mu_sum = mu.sum(dim=1)  # shape [batch, n]
            profit_modified = c + mu_sum     # shape [batch, n]
            loss = torch.sum(profit_modified * (X1 - X1p), dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()
        
        if time_monitoring:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            train_time += epoch_duration
        
        mean_loss = total_loss / len(dataloader_train)
        if time_monitoring:
            total_grad_norm = total_grad_norm ** 0.5
            current_lr = get_learning_rate(optimizer)            
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration, "train_time": train_time, "grad_norm": total_grad_norm, "lr": current_lr})

        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")
        with torch.no_grad():
            model.eval()
            total_regret = 0
            total_count = 0
            relat_regrets = []
            for z, c, x, _, _ in dataloader_test:
                z, c, x = [t.to(device) for t in (z, c, x)]
                c_hat = model(z)  # prédiction des coûts [batch, n]
                
                for i in range(z.size(0)):
                    model_i = knapsackModel(weights=weights, capacity=capacities)
                    c_numpy = c_hat[i].detach().cpu().numpy()
                    model_i.setObj(c_numpy)
                    x_hat_np, _ = model_i.solve()

                    x_true = x[i].to(dtype=torch.float32, device=device)
                    c_true = c[i].to(dtype=torch.float32, device=device)

                    x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32,
                                            device=device)

                    relat_regret = torch.dot(c_true, x_true - x_hat_tensor)/torch.dot(c_true, x_true)
                    relat_regrets.append(relat_regret)

            # Calcul de la norme de la différence entre mu et mu_tilde
            mu_data = dataloader_train.dataset.tensors[4].to(device)
            mu_diff = torch.norm(mu_data - mu_global, p='fro').item()  
            relat_regrets = torch.stack(relat_regrets)
                
        mean_relat_regret = relat_regrets.mean().item()
        std_relat_regret = relat_regrets.std().item()
        if run is not None:
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "Mean relative regret": mean_relat_regret, "Std relative regret": std_relat_regret, "train_time": train_time})
        if verbose:
            print(f"Eval Epoch {epoch} | Regret relatif moyen : {mean_relat_regret,:.4f}")
            
        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Limite de temps atteinte, arrêt de l'entraînement.")
                return
            
    if run is not None:   
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})