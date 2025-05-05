import torch
from torch import nn
import numpy as np
from pyepo.model.grb import portfolioModel
from pyepo.func import implicitMLE, SPOPlus


from my_solver import Solveur_lin

from opti_X_mu import Optimization_X_mu_portfolio
from joblib import Parallel, delayed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomMLP(nn.Module):
    '''
    Classe pour construire un modèle MLP personalisable
    '''
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
    '''Retourne le learning rate de l'optimiseur'''
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, method, run, dataloader_train, dataloader_test, optimizer, scheduler, cov, gamma, epochs=20, 
          IMLE_n_samples=10, IMLE_sigma=1.0, IMLE_lambd=10, IMLE_two_sides=False, IMLE_processes=1,
          SPO_solve_ratio = 1, SPO_reduction = 'mean', SPO_processes = 1,
          verbose=False):
    """
        Entraînement du modèle avec i-MLE classique
        model: modèle prédictif des profits
        run: wandb.run pour l'enregistrement des résultats
        dataloader: DataLoader avec (z, c, x, X1*(c), mu(c))
        optimizer: optimiseur PyTorch
        scheduler: planificateur d'apprentissage
        cov : matrice de covariance [n, n]
        gamma : niveau de risque
        epochs: nombre d’époques d'entraînement
        IMLE_n_samples: nombre d'échantillons pour i-MLE
        IMLE_sigma: sigma pour i-MLE
        IMLE_lambd: lambda pour i-MLE
        IMLE_two_sides: bool : Si True, utilise une perturbation à deux côtés
        IMLE_processes: int : Nombre de processus pour i-MLE
        verbose: bool : Si True, affiche les détails de l'entraînement
    """
    if run is not None:
        import time
        start_time = time.time()
        train_time = 0

    # Pour la résolution directe
    solver = portfolioModel(num_assets=cov.shape[0], covariance=cov, gamma=gamma) 
    # i-MLE avec solveur exact multi-contrainte
    if method == "imle":   
        gen = implicitMLE(solver, n_samples=IMLE_n_samples, sigma=IMLE_sigma, lambd=IMLE_lambd,
                        two_sides=IMLE_two_sides, processes=IMLE_processes)
    elif method == "spo":
        gen = SPOPlus(solver, solve_ratio = SPO_solve_ratio, reduction = SPO_reduction, processes = SPO_processes)

    for epoch in range(epochs):
        if run is not None:
            epoch_start_time = time.time()
        model.train()
        total_loss = 0
        total_grad_norm = 0.0
        
        for z, c, x, X1, mu in dataloader_train:
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            cp = model(z)

            if method == "imle":
                # Résolution avec la méthode choisie
                xp = gen(cp).to(device)
                # Regret = c · (w - wp)
                loss = torch.sum(c * (x - xp), dim=1).mean()
            elif method == "spo":
                true_cost = c.detach()
                true_sol = x.detach()
                obj_opt = (true_cost*true_sol).sum(dim=1)  
                loss = gen(cp, true_cost, true_sol, obj_opt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2

            total_loss += loss.item()
            
        if scheduler is not None:
            scheduler.step()
        
        mean_loss = total_loss / len(dataloader_train)
        if run is not None:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            total_grad_norm = total_grad_norm ** 0.5
            current_lr = get_learning_rate(optimizer)
            train_time += epoch_duration
            
            
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration, "train_time": train_time, "grad_norm": total_grad_norm, "lr": current_lr})

        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")
            
        with torch.no_grad():
            model.eval()
            total_regret = 0
            total_count = 0
            for z, c, x, _, _ in dataloader_test:
                z, c, x = [t.to(device) for t in (z, c, x)]
                c_hat = model(z)  # prédiction des coûts [batch, n]
                batch_regrets = []

                for i in range(z.size(0)):
                    solver_i = portfolioModel(num_assets=cov.shape[0], covariance=cov, gamma=gamma) 
                    c_numpy = c_hat[i].detach().cpu().numpy()
                    solver_i.setObj(c_numpy)
                    x_hat_np, _ = solver_i.solve()

                    x_true = x[i].to(dtype=torch.float32, device=device)
                    r_true = c[i].to(dtype=torch.float32, device=device)

                    x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32,
                                            device=device)

                    regret = torch.dot(r_true, x_true - x_hat_tensor)
                    batch_regrets.append(regret)

                batch_regrets = torch.stack(batch_regrets)
                total_regret += batch_regrets.sum().item()
                total_count += z.size(0)
                
        mean_regret = total_regret / total_count
        if run is not None:
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "regret": mean_regret, "train_time": train_time})
        if verbose:
            print(f"Eval Epoch {epoch+1} | Regret moyen : {mean_regret:.4f}")
            
    if run is not None:   
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})


def train_LD(model, method, run, dataloader_train, dataloader_test, optimizer, scheduler, cov, gamma, epochs=20, 
          IMLE_n_samples=10, IMLE_sigma=1.0, IMLE_lambd=10, IMLE_two_sides=False, IMLE_processes=1,
          SPO_solve_ratio = 1, SPO_reduction = 'mean', SPO_processes = 1,
          verbose=False):
    """
        Entraînement du modèle avec la décomposition lagrangienne
        model: modèle prédictif des profits
        run: wandb.run pour l'enregistrement des résultats
        dataloader: DataLoader avec (z, c, x, X1*(c), mu(c))
        optimizer: optimiseur PyTorch
        scheduler: planificateur d'apprentissage
        cov : matrice de covariance [n, n]
        gamma : niveau de risque
        epochs: nombre d’époques d'entraînement
        IMLE_n_samples: nombre d'échantillons pour i-MLE
        IMLE_sigma: sigma pour i-MLE
        IMLE_lambd: lambda pour i-MLE
        IMLE_two_sides: bool : Si True, utilise une perturbation à deux côtés
        IMLE_processes: int : Nombre de processus pour i-MLE
        verbose: bool : Si True, affiche les détails de l'entraînement
    """
    if run is not None:
        import time
        start_time = time.time()
        train_time = 0
    
    # Créer un solveur i-MLE avec les mu du batch
    solver = Solveur_lin(cov.shape[0])

    if method == "imle":
        gen = implicitMLE(solver, n_samples=IMLE_n_samples, sigma=IMLE_sigma, lambd=IMLE_lambd,
                       two_sides=IMLE_two_sides, processes=IMLE_processes)
    elif method == "spo":
        gen = SPOPlus(solver, solve_ratio = SPO_solve_ratio, reduction = SPO_reduction, processes = SPO_processes)

    for epoch in range(epochs):
        if run is not None:
            epoch_start_time = time.time()
        model.train()
        total_loss = 0
        total_grad_norm = 0.0

        for z, c, x, X1, mu in dataloader_train: # x vrai solution et X1 solution avec mu(c)
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            c_hat = model(z)  # prédiction des profits ĉ

            if method == "imle":
                # Résolution avec i-MLE
                X1p = gen(c_hat + mu).to(device)   # x̂ obtenu avec solve_main_problem
                # (c + mu_i) · (w - x̂)
                profit_modified = c + mu # shape [batch, n]
                loss = torch.sum(profit_modified * (X1 - X1p), dim=1).mean()
            elif method == "spo":
                true_cost = c.detach()+ mu.detach()
                true_sol = X1.detach()
                obj_opt = (true_cost*true_sol).sum(dim=1)  
                loss = gen(c_hat + mu, true_cost, true_sol, obj_opt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()


        mean_loss = total_loss / len(dataloader_train)
        if run is not None:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            total_grad_norm = total_grad_norm ** 0.5
            current_lr = get_learning_rate(optimizer)

            train_time += epoch_duration
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration, "train_time": train_time, "grad_norm": total_grad_norm, "lr": current_lr})
        
        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")

        
        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                total_regret = 0
                total_count = 0
                for z, c, x, _, _ in dataloader_test:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # prédiction des coûts [batch, n]
                    batch_regrets = []

                    for i in range(z.size(0)):
                        solver_i = portfolioModel(num_assets=cov.shape[0], covariance=cov, gamma=gamma) 
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        solver_i.setObj(c_numpy)
                        x_hat_np, _ = solver_i.solve()

                        x_true = x[i].to(dtype=torch.float32, device=device)
                        r_true = c[i].to(dtype=torch.float32, device=device)

                        x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32,
                                                device=device)

                        regret = torch.dot(r_true, x_true - x_hat_tensor)
                        batch_regrets.append(regret)

                    batch_regrets = torch.stack(batch_regrets)
                    total_regret += batch_regrets.sum().item()
                    total_count += z.size(0)
                    
            mean_regret = total_regret / total_count
            if run is not None:
                # Enregistrement des résultats dans wandb
                run.log({"epoch": epoch, "regret": mean_regret, "train_time": train_time})
            if verbose:
                print(f"Eval Epoch {epoch+1} | Regret moyen : {mean_regret:.4f}")
            
    if run is not None:   
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})

def train_SG(model, method, run, dataloader_train, dataloader_test, optimizer, scheduler, cov, gamma, epochs=20, 
        IMLE_n_samples=10, IMLE_sigma=1.0, IMLE_lambd=10, IMLE_two_sides=False, IMLE_processes=1,
        SPO_solve_ratio = 1, SPO_reduction = 'mean', SPO_processes = 1,
        verbose=False, step_mu=10, n_iter_mu = 100):
    """
        Entraînement du modèle avec la décomposition lagrangienne
        model: modèle prédictif des profits
        run: wandb.run pour l'enregistrement des résultats
        dataloader: DataLoader avec (z, c, x, X1*(c), mu(c))
        optimizer: optimiseur PyTorch
        scheduler: planificateur d'apprentissage
        epochs: nombre d’époques d'entraînement
        IMLE_n_samples: nombre d'échantillons pour i-MLE
        IMLE_sigma: sigma pour i-MLE
        IMLE_lambd: lambda pour i-MLE
        IMLE_two_sides: bool : Si True, utilise une perturbation à deux côtés
        IMLE_processes: int : Nombre de processus pour i-MLE
        verbose: bool : Si True, affiche les détails de l'entraînement
    """
    if run is not None:
        import time
        start_time = time.time()
        train_time = 0

    n_stocks = cov.shape[0]

    mu_global = torch.ones(len(dataloader_train.dataset), n_stocks, device=device, dtype = torch.float32)
    # Créer un solveur i-MLE avec les mu du batch
    solver = Solveur_lin(cov.shape[0])

    if method == "imle":
        gen = implicitMLE(solver, n_samples=IMLE_n_samples, sigma=IMLE_sigma, lambd=IMLE_lambd,
                two_sides=IMLE_two_sides, processes=IMLE_processes)
    elif method == "spo":
        gen = SPOPlus(solver, solve_ratio = SPO_solve_ratio, reduction = SPO_reduction, processes = SPO_processes)

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
                mu_list = Parallel(n_jobs=-1)(delayed(optimize_single_instance)(c_hat[i].detach().cpu().numpy(), cov, gamma, n_stocks, n_iter_mu, mu_tilde[i].detach().cpu().numpy()) for i in range(z.size(0)))
                mu_np = np.stack(mu_list)
                mu_tilde = torch.tensor(mu_np, device=device, dtype=torch.float32)
                mu_global[idx] = mu_tilde

            if method == "imle":
                # Résolution avec i-MLE
                X1p = gen(c_hat + mu_tilde).to(device)   # x̂ obtenu avec solve_main_problem
                # (c + mu) · (w - x̂)
                profit_modified = c + mu    # shape [batch, n]
                loss = torch.sum(profit_modified * (X1 - X1p), dim=1).mean()
            elif method == "spo":
                true_cost = c.detach()+ mu.detach()
                true_sol = X1.detach()
                obj_opt = (true_cost*true_sol).sum(dim=1)  
                loss = gen(c_hat + mu_tilde, true_cost, true_sol, obj_opt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2

            total_loss += loss.item()

        if scheduler is not None:
            scheduler.step()


        mean_loss = total_loss / len(dataloader_train)
        if run is not None:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            total_grad_norm = total_grad_norm ** 0.5
            current_lr = get_learning_rate(optimizer)

            train_time += epoch_duration
            
            if epoch % step_mu == 0:
                mu_data = dataloader_train.dataset.tensors[4].to(device)
                norm_mu_diff = torch.mean(torch.norm(mu_data - mu_global, dim=1, p='fro')).item()
            # Enregistrement des résultats dans wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration, "train_time": train_time, "grad_norm": total_grad_norm, "lr": current_lr, "mu_diff": norm_mu_diff})
        
        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")

                
        if epoch % 5 == 0:
            with torch.no_grad():
                model.eval()
                total_regret = 0
                total_count = 0
                for z, c, x, _, _ in dataloader_test:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # prédiction des coûts [batch, n]
                    batch_regrets = []

                    for i in range(z.size(0)):
                        solver_i = portfolioModel(num_assets=cov.shape[0], covariance=cov, gamma=gamma) 
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        solver_i.setObj(c_numpy)
                        x_hat_np, _ = solver_i.solve()

                        x_true = x[i].to(dtype=torch.float32, device=device)
                        r_true = c[i].to(dtype=torch.float32, device=device)

                        x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32,
                                                device=device)

                        regret = torch.dot(r_true, x_true - x_hat_tensor)
                        batch_regrets.append(regret)

                    batch_regrets = torch.stack(batch_regrets)
                    total_regret += batch_regrets.sum().item()
                    total_count += z.size(0)

            mean_regret = total_regret / total_count


            if run is not None:
                # Enregistrement des résultats dans wandb
                run.log({"epoch": epoch, "regret": mean_regret, "train_time": train_time, "mu_diff_norm": mu_diff})
            if verbose:
                print(f"Eval Epoch {epoch+1} | Regret moyen : {mean_regret:.4f} | ‖μ_global - μ_data‖_F : {mu_diff:.4f}")
            
    if run is not None:   
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})
        
        
def optimize_single_instance(c_i, cov, gamma, num_item, num_iter, mu0):
    optimizer = Optimization_X_mu_portfolio(
        num_item=num_item,
        c=c_i,
        cov=cov,
        gamma=gamma,
    )
    optimizer.optim_mu(mu0 = mu0, max_iter=num_iter)
    return optimizer.get_mu()