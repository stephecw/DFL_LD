import time
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_MSE(model, eval_solver, dataloader_train, dataloader_eval, optimizer, scheduler,
          epochs, time_limit, eval_freq,
          run, verbose=False, patience=10, min_delta=1e-6, device = device):
    """
    Training a PFL-model by minimizing the MSE.

    Args:
        model: ML model to train
        eval_solver: solver for the problem, used during eval
        run: wandb.run for logging results
        dataloader_train: DataLoader for training (z, c, x, X1*(c), mu(c))
        dataloader_eval: DataLoader for eval (z, c, x, X1*(c), mu(c))
        optimizer: PyTorch optimizer for training
        scheduler: PyTorch scheduler
        epochs: max number of training epochs
        time_limit: timeout on training time
        eval_freq: frequency of evaluation (in epochs)
        run: wandb logfile
        verbose: bool: If True, print training info
    """

    monitoring = run is not None or time_limit is not None
    best_relat_regret = float("inf")
    epochs_no_improvement = 0
    best_model_state = None
    best_epoch = 0


    eval_solver = eval_solver
    criterion = nn.MSELoss()

    if monitoring:
        start_time = time.time()
        train_time = 0

    for epoch in range(epochs):
        if monitoring:
            epoch_start_time = time.time()
        ## Training step ##
        model.train()
        total_loss = 0
        total_grad_norm = 0.0
        for z, c, x, _, _ in dataloader_train:
            z, c, x = [t.to(device) for t in (z, c, x)]
            c_hat = model(z)
            loss = criterion(c_hat, c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_loss += loss.item()

        mean_loss = total_loss / len(dataloader_train)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()

        if monitoring:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            train_time += epoch_duration
        if run is not None:
            total_grad_norm = total_grad_norm ** 0.5
            current_lr = get_learning_rate(optimizer)
            # Log results in wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration,
                    "train_time": train_time, "grad_norm": total_grad_norm, "lr": current_lr})
        if verbose:
            print(f"Epoch {epoch} | loss: {mean_loss:.4f}")

        ## evaluation step (if needed)##
        if epoch % eval_freq == 0:

            with torch.no_grad():
                model.eval()
                relat_regrets = []
                for z, c, x, _, _ in dataloader_eval:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # cost prediction [batch, n]

                    for i in range(z.size(0)):
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        eval_solver.setObj(c_numpy)
                        x_hat_np, _ = eval_solver.solve()
                        x_true = x[i].to(dtype=torch.float32, device=device)
                        c_true = c[i].to(dtype=torch.float32, device=device)
                        x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32, device=device)
                        # Compute regret
                        relat_regret = torch.dot(c_true, x_true - x_hat_tensor)/torch.dot(c_true, x_true)
                        relat_regrets.append(relat_regret)

                relat_regrets = torch.stack(relat_regrets)
                mean_relat_regret = relat_regrets.mean().item()
                std_relat_regret = relat_regrets.std().item()
                if run is not None:
                    # Log results in wandb
                    run.log({"epoch": epoch, "Mean relative regret": mean_relat_regret,
                            "Std relative regret": std_relat_regret, "train_time": train_time})
                if verbose:
                    print(f"Eval Epoch {epoch} | Mean relative regret: {mean_relat_regret:.4f}")
                
                # Early stopping
                if mean_relat_regret < best_relat_regret - min_delta:
                    best_relat_regret = mean_relat_regret
                    epochs_no_improvement = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                        break

        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Time limit reached, stopping training.")
                break

    if best_model_state is not None:
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    if run is not None:
        total_duration = time.time() - start_time
        run.log({
            "total_duration": total_duration,
            "best_epoch": best_epoch,
            "best_relat_regret": best_relat_regret
        })

def train_classic(model, diff_method, eval_solver, dataloader_train, dataloader_eval, optimizer, scheduler,
          epochs, time_limit, eval_freq,
          run, verbose=False, patience=10, min_delta=1e-6, device = device):
    """
    Training a DFL-model by minimizing classical regret loss.

    Args:
        model: ML model to train
        diff_method: DFL technique used to compute loss gradient
        eval_solver: solver for the problem, used during eval
        run: wandb.run for logging results
        dataloader_train: DataLoader for training (z, c, x, X1*(c), mu(c))
        dataloader_eval: DataLoader for eval (z, c, x, X1*(c), mu(c))
        optimizer: PyTorch optimizer for training
        scheduler: PyTorch scheduler
        epochs: max number of training epochs
        time_limit: timeout on training time
        eval_freq: frequency of evaluation (in epochs)
        run: wandb logfile
        verbose: bool: If True, print training info
    """

    monitoring = run is not None or time_limit is not None
    best_relat_regret = float("inf")
    epochs_no_improvement = 0
    best_model_state = None
    best_epoch = 0

    diff = diff_method
    eval_solver = eval_solver

    if monitoring:
        start_time = time.time()
        train_time = 0

    for epoch in range(epochs):
        if monitoring:
            epoch_start_time = time.time()

        ## Training step ##
        model.train()
        total_loss = 0
        total_grad_norm = 0.0
        for z, c, x, _, _ in dataloader_train:
            z, c, x = [t.to(device) for t in (z, c, x)]
            c_hat = model(z)
            loss = diff(c_hat, c, x).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_loss += loss.item()

        mean_loss = total_loss / len(dataloader_train)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()



        if monitoring:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            train_time += epoch_duration
        if run is not None:
            total_grad_norm = total_grad_norm ** 0.5
            current_lr = get_learning_rate(optimizer)
            # Log results in wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration,
                    "train_time": train_time, "grad_norm": total_grad_norm, "lr": current_lr})
        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")

        ## evaling step (if needed)##
        if epoch % eval_freq == 0:
            with torch.no_grad():
                model.eval()
                relat_regrets = []
                for z, c, x, _, _ in dataloader_eval:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # cost prediction [batch, n]

                    for i in range(z.size(0)):
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        eval_solver.setObj(c_numpy)
                        x_hat_np, _ = eval_solver.solve()
                        x_true = x[i].to(dtype=torch.float32, device=device)
                        c_true = c[i].to(dtype=torch.float32, device=device)
                        x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32, device=device)
                        # Compute regret
                        relat_regret = torch.dot(c_true, x_true - x_hat_tensor)/torch.dot(c_true, x_true)
                        relat_regrets.append(relat_regret)

                relat_regrets = torch.stack(relat_regrets)
                mean_relat_regret = relat_regrets.mean().item()
                std_relat_regret = relat_regrets.std().item()
                if run is not None:
                    # Log results in wandb
                    run.log({"epoch": epoch, "Mean relative regret": mean_relat_regret,
                            "Std relative regret": std_relat_regret, "train_time": train_time})
                if verbose:
                    print(f"Eval Epoch {epoch} | Mean relative regret: {mean_relat_regret:.4f}")

                # Early stopping
                if mean_relat_regret < best_relat_regret - min_delta:
                    best_relat_regret = mean_relat_regret
                    epochs_no_improvement = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                        break

        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Time limit reached, stopping training.")
                return

    if best_model_state is not None:
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        
    if run is not None:
        total_duration = time.time() - start_time
        run.log({
            "total_duration": total_duration,
            "best_epoch": best_epoch,
            "best_relat_regret": best_relat_regret
        })

def train_LD(model, diff_method, eval_solver, dataloader_train, dataloader_eval, optimizer, scheduler,
          epochs, time_limit, eval_freq,
          run, verbose=False, patience=10, min_delta=1e-6, device = device):
    """
    Training a DFL-model by minimizing LD loss.

    Args:
        model: ML model to train
        diff_method: DFL technique used to compute loss gradient
        eval_solver: solver for the problem, used during eval
        run: wandb.run for logging results
        dataloader_train: DataLoader for training (z, c, x, X1*(c), mu(c))
        dataloader_eval: DataLoader for eval (z, c, x, X1*(c), mu(c))
        optimizer: PyTorch optimizer for training
        scheduler: PyTorch scheduler
        epochs: max number of training epochs
        time_limit: timeout on training time
        eval_freq: frequency of evaluation (in epochs)
        run: wandb logfile
        verbose: bool: If True, print training info
    """

    monitoring = run is not None or time_limit is not None
    best_relat_regret = float("inf")
    epochs_no_improvement = 0
    best_model_state = None
    best_epoch = 0

    diff = diff_method
    eval_solver = eval_solver

    if monitoring:
        start_time = time.time()
        train_time = 0

    for epoch in range(epochs):
        if monitoring:
            epoch_start_time = time.time()

        ## Training step ##
        model.train()
        total_loss = 0
        total_grad_norm = 0.0
        for z, c, x, X1, mu in dataloader_train:
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            c_hat = model(z)
            mu_sum = torch.sum(mu, dim=1) # Shape (batch_size, num_item)
            loss = diff(c_hat + mu_sum, c + mu_sum, X1).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2
            total_loss += loss.item()

        mean_loss = total_loss / len(dataloader_train)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()

        if monitoring:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            train_time += epoch_duration
        if run is not None:
            total_grad_norm = total_grad_norm ** 0.5
            current_lr = get_learning_rate(optimizer)
            # Log results in wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration,
                    "train_time": train_time, "grad_norm": total_grad_norm, "lr": current_lr})
        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")

        ## evaling step (if needed)##
        if epoch % eval_freq == 0:
            with torch.no_grad():
                model.eval()
                relat_regrets = []
                for z, c, x, _, _ in dataloader_eval:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # cost prediction [batch, n]

                    for i in range(z.size(0)):
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        eval_solver.setObj(c_numpy)
                        x_hat_np, _ = eval_solver.solve()
                        x_true = x[i].to(dtype=torch.float32, device=device)
                        c_true = c[i].to(dtype=torch.float32, device=device)
                        x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32, device=device)
                        # Compute regret
                        relat_regret = torch.dot(c_true, x_true - x_hat_tensor)/torch.dot(c_true, x_true)
                        relat_regrets.append(relat_regret)

                relat_regrets = torch.stack(relat_regrets)
                mean_relat_regret = relat_regrets.mean().item()
                std_relat_regret = relat_regrets.std().item()
                if run is not None:
                    # Log results in wandb
                    run.log({"epoch": epoch, "Mean relative regret": mean_relat_regret,
                            "Std relative regret": std_relat_regret, "train_time": train_time})
                if verbose:
                    print(f"Eval Epoch {epoch} | Mean relative regret: {mean_relat_regret:.4f}")
                
                # Early stopping
                if mean_relat_regret < best_relat_regret - min_delta:
                    best_relat_regret = mean_relat_regret
                    epochs_no_improvement = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                        break

        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Time limit reached, stopping training.")
                return

    if best_model_state is not None:
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    if run is not None:
        total_duration = time.time() - start_time
        run.log({
            "total_duration": total_duration,
            "best_epoch": best_epoch,
            "best_relat_regret": best_relat_regret
        })

def train_SG(model, diff_method, eval_solver, dataloader_train, dataloader_eval, optimizer, scheduler,
          epochs, time_limit, eval_freq,
          step_mu, num_iter_mu, optimizer_mu,
          mu_global0,
          run, verbose=False, patience=10, min_delta=1e-6, device = device):
    """
    Training a DFL-model by minimizing LD loss, with adaptive mu.

    Args:
        model: ML model to train
        diff_method: DFL technique used to compute loss gradient
        eval_solver: solver for the problem, used during eval
        run: wandb.run for logging results
        dataloader_train: DataLoader for training (z, c, x, X1*(c), mu(c))
        dataloader_eval: DataLoader for eval (z, c, x, X1*(c), mu(c))
        optimizer: PyTorch optimizer for training
        scheduler: PyTorch scheduler
        epochs: max number of training epochs
        time_limit: timeout on training time
        num_items: number of items
        dim: number of constraints
        eval_freq: frequency of evaling (in epochs)
        step_mu: frequency of updating mu (in epochs)
        num_iter_mu: number of sub-gradient descent steps when updating mu
        optimizer_mu: optimizer object to update mu
        num_items: number of items
        dim: number of constraints
        run: wandb logfile
        verbose: bool: If True, print training info
    """

    monitoring = run is not None or time_limit is not None
    best_relat_regret = float("inf")
    epochs_no_improvement = 0
    best_model_state = None
    best_epoch = 0
    diff = diff_method
    eval_solver = eval_solver

    if monitoring:
        start_time = time.time()
        train_time = 0

    mu_global = mu_global0

    for epoch in range(epochs):
        if monitoring:
            epoch_start_time = time.time()

        ## Training step ##
        model.train()
        total_loss = 0
        total_grad_norm = 0.0
        for batch_idx, (z, c, x, X1, mu) in enumerate(dataloader_train):
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            c_hat = model(z)  # prediction ĉ
            idx = (batch_idx * dataloader_train.batch_size + torch.arange(z.size(0), device=device))
            mu_tilde = mu_global[idx] # select mu associated with the batch

            # Update mu_global
            if epoch % step_mu == 0:
                optimizer_mu.optim_mu(c_batch=c_hat.detach(),verbose=False, max_iter=num_iter_mu, mu_init=mu_tilde)
                mu_tilde = optimizer_mu.get_mu().detach()
                mu_global[idx] = mu_tilde

            # Forward and Backward pass
            mu_tilde_sum = mu_tilde.sum(dim=1) # Shape (batch_size, num_item)
            mu_sum = mu.sum(dim=1) # Shape (batch_size, num_item)
            loss = diff(c_hat + mu_tilde_sum, c + mu_sum, X1).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.norm().item() ** 2

        mean_loss = total_loss / len(dataloader_train)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()

        if monitoring:
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            train_time += epoch_duration
        if run is not None:
            total_grad_norm = total_grad_norm ** 0.5
            current_lr = get_learning_rate(optimizer)
            # Log results in wandb
            run.log({"epoch": epoch, "train_loss": mean_loss, "epoch_duration": epoch_duration,
                    "train_time": train_time, "grad_norm": total_grad_norm, "lr": current_lr})
        if verbose:
            print(f"Epoch {epoch+1} | loss: {mean_loss:.4f}")

        ## evaling step (if needed)##
        if epoch % eval_freq == 0:
            with torch.no_grad():
                model.eval()
                relat_regrets = []
                for z, c, x, _, _ in dataloader_eval:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # cost prediction [batch, n]

                    for i in range(z.size(0)):
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        eval_solver.setObj(c_numpy)
                        x_hat_np, _ = eval_solver.solve()
                        x_true = x[i].to(dtype=torch.float32, device=device)
                        c_true = c[i].to(dtype=torch.float32, device=device)
                        x_hat_tensor = torch.tensor(x_hat_np, dtype=torch.float32, device=device)
                        # Compute regret
                        relat_regret = torch.dot(c_true, x_true - x_hat_tensor)/torch.dot(c_true, x_true)
                        relat_regrets.append(relat_regret)

                # Compute difference between optimal mu*(c) and adaptive mu
                mu_data = dataloader_train.dataset.tensors[4].to(device)
                mu_diff = torch.norm(mu_data - mu_global, p='fro').item()

                relat_regrets = torch.stack(relat_regrets)
                mean_relat_regret = relat_regrets.mean().item()
                std_relat_regret = relat_regrets.std().item()
                if run is not None:
                    # Log results in wandb
                    run.log({"epoch": epoch, "Mean relative regret": mean_relat_regret,
                            "Std relative regret": std_relat_regret, "norm_diff_mu": mu_diff,
                            "train_time": train_time})
                if verbose:
                    print(f"Eval Epoch {epoch} | Mean relative regret: {mean_relat_regret:.4f} | norm_diff_mu: {mu_diff:.4f}")
                
                # Early stopping
                if mean_relat_regret < best_relat_regret - min_delta:
                    best_relat_regret = mean_relat_regret
                    epochs_no_improvement = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
                        if run is not None:
                            run.log({"training_status": "early_stopping"})
                        break

        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Time limit reached, stopping training.")
                if best_model_state is not None:
                    device = next(model.parameters()).device
                    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                    if run is not None:
                        total_duration = time.time() - start_time
                        run.log({
                            "total_duration": total_duration,
                            "best_epoch": best_epoch,
                            "best_relat_regret": best_relat_regret,
                            "training_status": "time_limit"
                        })
                return
    
    if best_model_state is not None:
        device = next(model.parameters()).device
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

    if run is not None:
        total_duration = time.time() - start_time
        run.log({
            "total_duration": total_duration,
            "best_epoch": best_epoch,
            "best_relat_regret": best_relat_regret
        })


def test(model, test_loader, eval_solver, device, run=None):
    """
    Compute mean and std of relative regret on a test set.

    Args:
        model       : trained nn.Module
        test_loader : DataLoader yielding (z, c, x, *_)
        eval_solver : solver with setObj()/solve() interface
        device      : 'cpu' or 'cuda'
        run         : optional wandb run for logging

    Returns:
        mean_relat, std_relat
    """
    model.eval()
    rel_regr = []

    with torch.no_grad():
        for z, c, x, *_ in test_loader:
            z, c, x = z.to(device), c.to(device), x.to(device)
            c_hat = model(z)

            for i in range(z.size(0)):
                eval_solver.setObj(c_hat[i].cpu().numpy())
                x_pred_np, _ = eval_solver.solve()

                x_true = x[i].float().cpu()
                c_true = c[i].float().cpu()
                x_pred = torch.tensor(x_pred_np, dtype=torch.float32)

                num = torch.dot(c_true, x_true - x_pred)
                den = torch.dot(c_true, x_true).clamp(min=1e-6)
                rel_regr.append((num / den).item())

    errs = torch.tensor(rel_regr)
    mean_relat = errs.mean().item()
    std_relat  = errs.std().item()

    print(f"Test relative regret: {mean_relat:.4f} ± {std_relat:.4f}")
    if run:
        run.log({
            "test/mean_relative_regret": mean_relat,
            "test/std_relative_regret": std_relat
        })

    return mean_relat, std_relat