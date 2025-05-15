import torch
from torch import nn
from pyepo.func import implicitMLE, SPOPlus

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

def train(model, optmodel_train, optmodel_eval, run, dataloader_train, dataloader_eval, optimizer, scheduler,
          epochs, time_limit,
          diff_method, diff_method_arg,
          verbose=False, patience= 10, min_delta= 1e-6):
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
    best_relat_regret = float("inf")
    epochs_no_improvement = 0
    best_model_state = None
    best_epoch = 0
    
    # i-MLE avec solveur exact multi-contrainte
    if diff_method == "IMLE":
        diff = implicitMLE(optmodel_train, **diff_method_arg)
    elif diff_method == "SPOPlus":
        diff = SPOPlus(optmodel_eval, **diff_method_arg)

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
            c_hat = model(z)
            if diff_method == "IMLE":
                x_hat = diff(c_hat).to(device)   # x̂ obtenu avec solve_main_problem
                loss = torch.sum(c * (x - x_hat), dim=1).mean()
            elif diff_method == "SPOPlus":
                loss = diff(-c_hat, -c, x.float(), (-c * x).sum(dim=1)).to(device)

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
        if diff_method != "SPOPlus" or epoch % 5 == 0: 
            with torch.no_grad():
                model.eval()
                total_regret = 0
                total_count = 0
                relat_regrets = []
                for z, c, x, _, _ in dataloader_eval:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # prédiction des coûts [batch, n]
                    
                    for i in range(z.size(0)):
                        model_i = optmodel_eval
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
                print(f"Eval Epoch {epoch} | Regret relatif moyen : {mean_relat_regret:.4f}")

            # Early stopping
            if mean_relat_regret < best_relat_regret - min_delta:
                best_relat_regret = mean_relat_regret
                epochs_no_improvement = 0
                best_model_state = { k: v.cpu().clone() for k,v in model.state_dict().items() }
                best_epoch = epoch
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement >= patience:
                    print(f"Arrêt précoce à l'époque {epoch} | Meilleur regret relatif : {best_relat_regret:.4f}")
                    break
                
        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Limite de temps atteinte, arrêt de l'entraînement.")
                return
            
    # Charger le meilleur état du modèle
    if best_model_state is not None:
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            
    if run is not None:   
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration, "best_epoch": best_epoch, "best_relat_regret": best_relat_regret})




def train_LD(model, solver_train, solver_eval, run, dataloader_train, dataloader_eval, optimizer, scheduler,
          epochs, time_limit,
          diff_method, diff_method_arg,
          verbose=False, patience= 10, min_delta= 1e-6):
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
    best_relat_regret = float("inf")
    epochs_no_improvement = 0
    best_model_state = None
    best_epoch = 0

    
    # Créer un solveur i-MLE avec les mu du batch
    if diff_method == "IMLE":
        diff = implicitMLE(solver_train, **diff_method_arg)
    elif diff_method == "SPOPlus":
        diff = SPOPlus(solver_train, **diff_method_arg)

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
            if diff_method == "IMLE":
                X1p = diff(c_hat + mu_sum).to(device)   # x̂ obtenu avec solve_main_problem
                # (c + sum mu_i for i ≥ 2) · (w - x̂)
                profit_modified = c + mu_sum     # shape [batch, n]
                loss = torch.sum(profit_modified * (X1 - X1p), dim=1).mean()
            elif diff_method == "SPOPlus":
                loss = diff(-c_hat - mu_sum, -c - mu_sum, X1.float(), (-(c + mu_sum) * X1).sum(dim=1)).to(device)
            
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
            
        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                total_regret = 0
                total_count = 0
                relat_regrets = []
                for z, c, x, _, _ in dataloader_eval:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # prédiction des coûts [batch, n]
                    
                    for i in range(z.size(0)):
                        model_i = solver_eval
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
                print(f"Eval Epoch {epoch} | Regret relatif moyen : {mean_relat_regret:.4f}")
            
            # Early stopping
            if mean_relat_regret < best_relat_regret - min_delta:
                best_relat_regret = mean_relat_regret
                epochs_no_improvement = 0
                best_model_state = { k: v.cpu().clone() for k,v in model.state_dict().items() }
                best_epoch = epoch
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement >= patience:
                    print(f"Arrêt précoce à l'époque {epoch} | Meilleur regret relatif : {best_relat_regret:.4f}")
                    break
                
        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Limite de temps atteinte, arrêt de l'entraînement.")
                return
    
    # Charger le meilleur état du modèle
    if best_model_state is not None:
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            
    if run is not None:   
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration, "best_epoch": best_epoch, "best_relat_regret": best_relat_regret})


def train_SG(model, solver_train, solver_eval, optimizer_mu, mu_init, run, dataloader_train, dataloader_eval, optimizer, scheduler,
          epochs, time_limit,
          diff_method, diff_method_arg,
          step_mu=10, num_iter_mu = 100,
          verbose=False, patience= 10, min_delta= 1e-6):
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
    best_relat_regret = float("inf")
    epochs_no_improvement = 0
    best_model_state = None
    best_epoch = 0
    
    # Créer un solveur i-MLE avec les mu du batch
    if diff_method == "IMLE":
        diff = implicitMLE(solver_train, **diff_method_arg)
    elif diff_method == "SPOPlus":
        diff = SPOPlus(solver_train, **diff_method_arg)

    mu_global = mu_init.clone().to(device)
    
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
                optimizer_mu = optimizer_mu
                optimizer_mu.optim_mu_batch(mu0 = mu_tilde.clone(), verbose=False, max_iter=num_iter_mu)
                mu_tilde = optimizer_mu.get_mu().detach()
                mu_global[idx] = mu_tilde


            # Résolution avec i-MLE
            mu_tilde_sum = mu_tilde.sum(dim=1)
            mu_sum = mu.sum(dim=1)
            if diff_method == "IMLE":
                X1p = diff(c_hat + mu_tilde_sum).to(device)   # x̂ obtenu avec solve_main_problem
                # (c + sum mu_i for i ≥ 2) · (w - x̂)
                profit_modified = c + mu_sum     # shape [batch, n]
                loss = torch.sum(profit_modified * (X1 - X1p), dim=1).mean()
            elif diff_method == "SPOPlus":
                loss = diff(-c_hat - mu_tilde_sum, -c + mu_sum, X1.float(), (-(c + mu_tilde_sum) * X1).sum(dim=1)).to(device)
            
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
        
        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                total_regret = 0
                total_count = 0
                relat_regrets = []
                for z, c, x, _, _ in dataloader_eval:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # prédiction des coûts [batch, n]
                    
                    for i in range(z.size(0)):
                        model_i = solver_eval
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
                print(f"Eval Epoch {epoch} | Regret relatif moyen : {mean_relat_regret:.4f}")
            
            # Early stopping
            if mean_relat_regret < best_relat_regret - min_delta:
                best_relat_regret = mean_relat_regret
                epochs_no_improvement = 0
                best_model_state = { k: v.cpu().clone() for k,v in model.state_dict().items() }
                best_epoch = epoch
            else:
                epochs_no_improvement += 1
                if epochs_no_improvement >= patience:
                    print(f"Arrêt précoce à l'époque {epoch} | Meilleur regret relatif : {best_relat_regret:.4f}")
                    break
                
        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Limite de temps atteinte, arrêt de l'entraînement.")
                return
    
    # Charger le meilleur état du modèle
    if best_model_state is not None:
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
            
    if run is not None:   
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration, "best_epoch": best_epoch, "best_relat_regret": best_relat_regret})