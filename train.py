import time
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_MSE(model, test_solver, dataloader_train, dataloader_test, optimizer, scheduler,
          epochs, time_limit, test_freq,
          run, verbose=False):
    """
    Training a PFL-model by minimizing the MSE.

    Args:
        model: ML model to train
        test_solver: solver for the problem, used during test
        run: wandb.run for logging results
        dataloader_train: DataLoader for training (z, c, x, X1*(c), mu(c))
        dataloader_test: DataLoader for test (z, c, x, X1*(c), mu(c))
        optimizer: PyTorch optimizer for training
        scheduler: PyTorch scheduler
        epochs: max number of training epochs
        time_limit: timeout on training time
        test_freq: frequency of testing (in epochs)
        run: wandb logfile
        verbose: bool: If True, print training info
    """

    monitoring = run is not None or time_limit is not None

    test_solver = test_solver
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
        if scheduler is not None:
            scheduler.step()

        mean_loss = total_loss / len(dataloader_train)

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

        ## Testing step (if needed)##
        if epoch % test_freq == 0:
            with torch.no_grad():
                model.eval()
                relat_regrets = []
                for z, c, x, _, _ in dataloader_test:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # cost prediction [batch, n]

                    for i in range(z.size(0)):
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        test_solver.setObj(c_numpy)
                        x_hat_np, _ = test_solver.solve()
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

        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Time limit reached, stopping training.")
                return

    if run is not None:
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})

def train_classic(model, diff_method, test_solver, dataloader_train, dataloader_test, optimizer, scheduler,
          epochs, time_limit, test_freq,
          run, verbose=False):
    """
    Training a DFL-model by minimizing classical regret loss.

    Args:
        model: ML model to train
        diff_method: DFL technique used to compute loss gradient
        test_solver: solver for the problem, used during test
        run: wandb.run for logging results
        dataloader_train: DataLoader for training (z, c, x, X1*(c), mu(c))
        dataloader_test: DataLoader for test (z, c, x, X1*(c), mu(c))
        optimizer: PyTorch optimizer for training
        scheduler: PyTorch scheduler
        epochs: max number of training epochs
        time_limit: timeout on training time
        test_freq: frequency of testing (in epochs)
        run: wandb logfile
        verbose: bool: If True, print training info
    """

    monitoring = run is not None or time_limit is not None

    diff = diff_method
    test_solver = test_solver

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
        if scheduler is not None:
            scheduler.step()

        mean_loss = total_loss / len(dataloader_train)

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

        ## Testing step (if needed)##
        if epoch % test_freq == 0:
            with torch.no_grad():
                model.eval()
                relat_regrets = []
                for z, c, x, _, _ in dataloader_test:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # cost prediction [batch, n]

                    for i in range(z.size(0)):
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        test_solver.setObj(c_numpy)
                        x_hat_np, _ = test_solver.solve()
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

        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Time limit reached, stopping training.")
                return

    if run is not None:
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})

def train_LD(model, diff_method, test_solver, dataloader_train, dataloader_test, optimizer, scheduler,
          epochs, time_limit, test_freq,
          run, verbose=False):
    """
    Training a DFL-model by minimizing LD loss.

    Args:
        model: ML model to train
        diff_method: DFL technique used to compute loss gradient
        test_solver: solver for the problem, used during test
        run: wandb.run for logging results
        dataloader_train: DataLoader for training (z, c, x, X1*(c), mu(c))
        dataloader_test: DataLoader for test (z, c, x, X1*(c), mu(c))
        optimizer: PyTorch optimizer for training
        scheduler: PyTorch scheduler
        epochs: max number of training epochs
        time_limit: timeout on training time
        test_freq: frequency of testing (in epochs)
        run: wandb logfile
        verbose: bool: If True, print training info
    """

    monitoring = run is not None or time_limit is not None

    diff = diff_method
    test_solver = test_solver

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
        if scheduler is not None:
            scheduler.step()

        mean_loss = total_loss / len(dataloader_train)

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

        ## Testing step (if needed)##
        if epoch % test_freq == 0:
            with torch.no_grad():
                model.eval()
                relat_regrets = []
                for z, c, x, _, _ in dataloader_test:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # cost prediction [batch, n]

                    for i in range(z.size(0)):
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        test_solver.setObj(c_numpy)
                        x_hat_np, _ = test_solver.solve()
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

        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Time limit reached, stopping training.")
                return

    if run is not None:
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})

def train_SG(model, diff_method, test_solver, dataloader_train, dataloader_test, optimizer, scheduler,
          epochs, time_limit, test_freq,
          step_mu, num_iter_mu, optimizer_mu,
          num_items, dim,
          run, verbose=False):
    """
    Training a DFL-model by minimizing LD loss, with adaptive mu.

    Args:
        model: ML model to train
        diff_method: DFL technique used to compute loss gradient
        test_solver: solver for the problem, used during test
        run: wandb.run for logging results
        dataloader_train: DataLoader for training (z, c, x, X1*(c), mu(c))
        dataloader_test: DataLoader for test (z, c, x, X1*(c), mu(c))
        optimizer: PyTorch optimizer for training
        scheduler: PyTorch scheduler
        epochs: max number of training epochs
        time_limit: timeout on training time
        num_items: number of items
        dim: number of constraints
        test_freq: frequency of testing (in epochs)
        step_mu: frequency of updating mu (in epochs)
        num_iter_mu: number of sub-gradient descent steps when updating mu
        optimizer_mu: optimizer object to update mu
        num_items: number of items
        dim: number of constraints
        run: wandb logfile
        verbose: bool: If True, print training info
    """

    monitoring = run is not None or time_limit is not None

    diff = diff_method
    test_solver = test_solver

    if monitoring:
        start_time = time.time()
        train_time = 0

    mu_global = torch.ones(len(dataloader_train.dataset), dim - 1, num_items, device=device, dtype=torch.float32)

    for epoch in range(epochs):
        if monitoring:
            epoch_start_time = time.time()

        ## Training step ##
        model.train()
        total_loss = 0
        total_grad_norm = 0.0
        for batch_idx, (z, c, x, X1, mu) in enumerate(dataloader_train):
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            c_hat = model(z)  # prediction ĉ
            idx = (batch_idx * dataloader_train.batch_size + torch.arange(z.size(0), device=device))
            mu_tilde = mu_global[idx] # select mu associated with the batch

            # Update mu_global
            if epoch % step_mu == 0:
                optimizer_mu.optim_mu(c_batch=c_hat.detach(), verbose=False, max_iter=num_iter_mu, mu_init=mu_tilde)
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
        if scheduler is not None:
            scheduler.step()

        mean_loss = total_loss / len(dataloader_train)

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

        ## Testing step (if needed)##
        if epoch % test_freq == 0:
            with torch.no_grad():
                model.eval()
                relat_regrets = []
                for z, c, x, _, _ in dataloader_test:
                    z, c, x = [t.to(device) for t in (z, c, x)]
                    c_hat = model(z)  # cost prediction [batch, n]

                    for i in range(z.size(0)):
                        c_numpy = c_hat[i].detach().cpu().numpy()
                        test_solver.setObj(c_numpy)
                        x_hat_np, _ = test_solver.solve()
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
                    print(f"Eval Epoch {epoch} | Mean relative regret: {mean_relat_regret:.4f}")

        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Time limit reached, stopping training.")
                return

    if run is not None:
        end_time = time.time()
        total_duration = end_time - start_time
        run.log({"total_duration": total_duration})
