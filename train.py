import time
import copy
import torch
import numpy as np
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def _compute_regret(c_hat_i, x_true_i, c_true_i, eval_solver):
    # run solver and compute one sample’s regret
    eval_solver.setObj(c_hat_i)
    x_hat, _ = eval_solver.solve()
    num = (c_true_i * (x_true_i - x_hat)).sum()
    den = max((c_true_i * x_true_i).sum(), 1e-6)
    return num / den

def train_MSE(model, eval_solver, dataloader_train, dataloader_eval, dataloader_test, optimizer, scheduler,
          epochs, time_limit, eval_freq, report_times,
          run, verbose=False, patience=None, min_delta=1e-6, device = device, n_jobs=-1):
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
    report_times = sorted(report_times)
    pending = {t:True for t in report_times}  # pending times to report
    num_test = len(dataloader_test.dataset)
    num_eval = len(dataloader_eval.dataset)
    result = torch.zeros((len(report_times),num_test), dtype=torch.float32, device=device)
    result_eval = torch.zeros((len(report_times),num_eval), dtype=torch.float32, device=device)
    monitoring = run is not None or time_limit is not None
    best_relat_regret = float("inf")
    epochs_no_improvement = 0
    best_model_state = None
    best_epoch = 0
    best_model = copy.deepcopy(model).to(device)


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
            print(f"Epoch {epoch} | loss: {mean_loss:.4f} | train time : {train_time:.4f}", flush=True)

        ## evaluation step (if needed)##
        if epoch % eval_freq == eval_freq-1:

            with torch.no_grad():
                model.eval()
                all_regrets = []

                for z, c, x, _, _ in dataloader_eval:
                    z = z.to(device)
                    c_hat = model(z)  # [batch, n]

                    # prepare the inputs for the helper
                    c_hat_np = c_hat.detach().cpu().numpy()   # shape [B,n]
                    x_true_cpu = x.float().cpu().numpy()              # shape [B,n], torch Tensor
                    c_true_cpu = c.float().cpu().numpy() 

                    for i in range(z.size(0)):
                        eval_solver.setObj(c_hat_np[i]) 
                        x_hat, _ = eval_solver.solve()
                        num = (c_true_cpu[i] * (x_true_cpu[i] - x_hat)).sum()
                        den = max((c_true_cpu[i] * x_true_cpu[i]).sum(), 1e-6)
                        rel_regret = num / den
                        all_regrets.append(rel_regret)

                mean_relat_regret = float(np.mean(all_regrets))
                std_relat_regret  = float(np.std(all_regrets))
                if run is not None:
                    # Log results in wandb
                    run.log({"epoch": epoch, "Mean relative regret": mean_relat_regret,
                            "Std relative regret": std_relat_regret, "train_time": train_time})
                if verbose:
                    print(f"Eval Epoch {epoch} | Mean relative regret: {mean_relat_regret:.4f}", flush=True)
                
                # Early stopping
                if mean_relat_regret < best_relat_regret - min_delta:
                    best_relat_regret = mean_relat_regret
                    epochs_no_improvement = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    best_model.load_state_dict(model.state_dict())
                elif patience is not None:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}", flush=True)
                        break
        
        for i,t in enumerate(report_times):
            if train_time >= t and pending[t] and t > 0:
                pending[t] = False
                best_model.eval()
                regrets= test(best_model, dataloader_test, eval_solver, device, run=None)
                regrets_eval = test(best_model, dataloader_eval, eval_solver, device, run=None)
                result[i] = torch.tensor(regrets, dtype=torch.float32, device=device)
                result_eval[i] = torch.tensor(regrets_eval, dtype=torch.float32, device=device)


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
    return result_eval,result

def train_classic(model, diff_method, eval_solver, dataloader_train, dataloader_eval, dataloader_test, optimizer, scheduler,
          epochs, time_limit, eval_freq, report_times,
          run, verbose=False, patience=None, min_delta=1e-6, device = device, n_jobs=-1):
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
    report_times = sorted(report_times)
    pending = {t:True for t in report_times}  # pending times to report
    num_test = len(dataloader_test.dataset)
    num_eval = len(dataloader_eval.dataset)
    result = torch.zeros((len(report_times),num_test), dtype=torch.float32, device=device)
    result_eval = torch.zeros((len(report_times),num_eval), dtype=torch.float32, device=device)
    monitoring = run is not None or time_limit is not None
    best_relat_regret = float("inf")
    epochs_no_improvement = 0
    best_model_state = None
    best_epoch = 0
    best_model = copy.deepcopy(model).to(device)

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
            print(f"Epoch {epoch} | loss: {mean_loss:.4f} | train time : {train_time:.4f}", flush=True)

        ## evaling step (if needed)##
        if epoch % eval_freq == eval_freq-1:
            with torch.no_grad():
                model.eval()
                all_regrets = []

                for z, c, x, _, _ in dataloader_eval:
                    z = z.to(device)
                    c_hat = model(z)  # [batch, n]

                    # prepare the inputs for the helper
                    c_hat_np = c_hat.detach().cpu().numpy()   # shape [B,n]
                    x_true_cpu = x.float().cpu().numpy()               # shape [B,n], torch Tensor
                    c_true_cpu = c.float().cpu().numpy()  

                    for i in range(z.size(0)):
                        eval_solver.setObj(c_hat_np[i]) 
                        x_hat, _ = eval_solver.solve()
                        num = (c_true_cpu[i] * (x_true_cpu[i] - x_hat)).sum()
                        den = max((c_true_cpu[i] * x_true_cpu[i]).sum(), 1e-6)
                        rel_regret = num / den
                        all_regrets.append(rel_regret)

                mean_relat_regret = float(np.mean(all_regrets))
                std_relat_regret  = float(np.std(all_regrets))
                if run is not None:
                    # Log results in wandb
                    run.log({"epoch": epoch, "Mean relative regret": mean_relat_regret,
                            "Std relative regret": std_relat_regret, "train_time": train_time})
                if verbose:
                    print(f"Eval Epoch {epoch} | Mean relative regret: {mean_relat_regret:.4f}", flush=True)

                # Early stopping
                if mean_relat_regret < best_relat_regret - min_delta:
                    best_relat_regret = mean_relat_regret
                    epochs_no_improvement = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    best_model.load_state_dict(model.state_dict())
                elif patience is not None:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}", flush=True)
                        break
        
        for i,t in enumerate(report_times):
            if train_time >= t and pending[t] and t > 0:
                pending[t] = False
                best_model.eval()
                regrets= test(best_model, dataloader_test, eval_solver, device, run=None)
                regrets_eval = test(best_model, dataloader_eval, eval_solver, device, run=None)
                result[i] = torch.tensor(regrets, dtype=torch.float32, device=device)
                result_eval[i] = torch.tensor(regrets_eval, dtype=torch.float32, device=device)

        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Time limit reached, stopping training.", flush=True)
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
    return result_eval,result

def train_LD(model, diff_method, eval_solver, dataloader_train, dataloader_eval, dataloader_test, optimizer, scheduler,
          epochs, time_limit, eval_freq, report_times,
          run, verbose=False, patience=None, min_delta=1e-6, device = device, n_jobs=-1, muloss=True, mains=[0], combine = "random"):
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
    assert len(mains)==len(dataloader_train), "mains must have the same length as the number of datasets"

    report_times = sorted(report_times)
    pending = {t:True for t in report_times}  # pending times to report
    num_test = len(dataloader_test.dataset)
    num_eval = len(dataloader_eval.dataset)
    result = torch.zeros((len(report_times),num_test), dtype=torch.float32, device=device)
    result_eval = torch.zeros((len(report_times),num_eval), dtype=torch.float32, device=device)
    monitoring = run is not None or time_limit is not None
    best_relat_regret = float("inf")
    epochs_no_improvement = 0
    best_model_state = None
    best_epoch = 0
    best_model = copy.deepcopy(model).to(device)

    diff = diff_method
    eval_solver = eval_solver

    if monitoring:
        start_time = time.time()
        train_time = 0

    
    mu_add = []
    for i in range(1,len(mains)):
        mu_add.append(dataloader_train[i].dataset.tensors[4].clone().to(device))

    X1_add = []
    for i in range(1,len(mains)):
        X1_add.append(dataloader_train[i].dataset.tensors[3].clone().to(device))

    for epoch in range(epochs):
        if monitoring:
            epoch_start_time = time.time()

        ## Training step ##
        model.train()
        total_loss = 0
        total_grad_norm = 0.0
        for batch_idx, (z, c, x, X1, mu) in enumerate(dataloader_train[0]):
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            c_hat = model(z)

            idx = (batch_idx * dataloader_train[0].batch_size + torch.arange(z.size(0), device=device))
            mu_batch_add = [mu_add[i][idx] for i in range(len(mu_add))] # select mu associated with the batch
            X1_batch_add = [X1_add[i][idx] for i in range(len(X1_add))]

            if combine == "sum":
                mu_sum = torch.sum(mu, dim=1) # Shape (batch_size, num_item)
                mu_sum2 = torch.sum(mu, dim=1) if muloss else 0
                loss = diff(c_hat + mu_sum, c + mu_sum2, X1).to(device)

                for i in range(1,len(mains)):
                    mu_batch_add_sum = torch.sum(mu_batch_add[i-1], dim=1)
                    mu_batch_add_sum2 = torch.sum(mu_batch_add[i-1], dim=1) if muloss else 0
                    loss += diff(c_hat + mu_batch_add_sum, c + mu_batch_add_sum2, X1_batch_add[i-1]).to(device)
            
            elif combine == "random":
                k = np.random.randint(0, len(mains))
                if k == 0:
                    mu_sum = torch.sum(mu, dim=1)
                    mu_sum2 = torch.sum(mu, dim=1) if muloss else 0
                    loss = diff(c_hat + mu_sum, c + mu_sum2, X1).to(device)
                else:
                    mu_batch_add_sum = torch.sum(mu_batch_add[k-1], dim=1)
                    mu_batch_add_sum2 = torch.sum(mu_batch_add[k-1], dim=1) if muloss else 0
                    loss = diff(c_hat + mu_batch_add_sum, c + mu_batch_add_sum2, X1_batch_add[k-1]).to(device)

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
            print(f"Epoch {epoch} | loss: {mean_loss:.4f} | train time : {train_time}", flush=True)

        ## evaling step (if needed)##
        if epoch % eval_freq == eval_freq-1:
            with torch.no_grad():
                model.eval()
                all_regrets = []

                for z, c, x, _, _ in dataloader_eval:
                    z = z.to(device)
                    c_hat = model(z)  # [batch, n]

                    # prepare the inputs for the helper
                    c_hat_np = c_hat.detach().cpu().numpy()   # shape [B,n]
                    x_true_cpu = x.float().cpu().numpy()               # shape [B,n], torch Tensor
                    c_true_cpu = c.float().cpu().numpy()  

                    for i in range(z.size(0)):
                        eval_solver.setObj(c_hat_np[i]) 
                        x_hat, _ = eval_solver.solve()
                        num = (c_true_cpu[i] * (x_true_cpu[i] - x_hat)).sum()
                        den = max((c_true_cpu[i] * x_true_cpu[i]).sum(), 1e-6)
                        rel_regret = num / den
                        all_regrets.append(rel_regret)

                mean_relat_regret = float(np.mean(all_regrets))
                std_relat_regret  = float(np.std(all_regrets))
                if run is not None:
                    # Log results in wandb
                    run.log({"epoch": epoch, "Mean relative regret": mean_relat_regret,
                            "Std relative regret": std_relat_regret, "train_time": train_time})
                if verbose:
                    print(f"Eval Epoch {epoch} | Mean relative regret: {mean_relat_regret:.4f}", flush=True)
                
                # Early stopping
                if mean_relat_regret < best_relat_regret - min_delta:
                    best_relat_regret = mean_relat_regret
                    epochs_no_improvement = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    best_model.load_state_dict(model.state_dict())
                elif patience is not None:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}", flush=True)
                        break

        for i,t in enumerate(report_times):
            if train_time >= t and pending[t] and t > 0:
                pending[t] = False
                best_model.eval()
                regrets= test(best_model, dataloader_test, eval_solver, device, run=None)
                regrets_eval = test(best_model, dataloader_eval, eval_solver, device, run=None)
                result[i] = torch.tensor(regrets, dtype=torch.float32, device=device)
                result_eval[i] = torch.tensor(regrets_eval, dtype=torch.float32, device=device)

        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Time limit reached, stopping training.", flush=True)
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
    return result_eval,result

def train_SG(model, diff_method, eval_solver, dataloader_train, dataloader_eval, dataloader_test, optimizer, scheduler,
          epochs, time_limit, eval_freq, report_times,
          step_mu, num_iter_mu, optimizer_mu,
          mu_global0,
          run, verbose=False, patience=None, min_delta=1e-6, device = device, n_jobs=-1, muloss=True, mains=[0], combine = "random"):
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
    assert len(mains)==len(dataloader_train), "mains must have the same length as the number of datasets"

    report_times = sorted(report_times)
    pending = {t:True for t in report_times}  # pending times to report
    num_test = len(dataloader_test.dataset)
    num_eval = len(dataloader_eval.dataset)
    result = torch.zeros((len(report_times),num_test), dtype=torch.float32, device=device)
    result_eval = torch.zeros((len(report_times),num_eval), dtype=torch.float32, device=device)
    monitoring = run is not None or time_limit is not None
    best_relat_regret = float("inf")
    epochs_no_improvement = 0
    best_model_state = None
    best_epoch = 0
    best_model = copy.deepcopy(model).to(device)

    diff = diff_method
    eval_solver = eval_solver

    if monitoring:
        start_time = time.time()
        train_time = 0

    mu_global = mu_global0
    mu_global_add = [mu_global.clone() for i in range(len(mains)-1)]

    if muloss:
        mu_add = []
        for i in range(1,len(mains)):
            mu_add.append(dataloader_train[i].dataset.tensors[4].clone().to(device))
        
        X1_add = []
        for i in range(1,len(mains)):
            X1_add.append(dataloader_train[i].dataset.tensors[3].clone().to(device))

    for epoch in range(epochs):
        if monitoring:
            epoch_start_time = time.time()

        ## Training step ##
        model.train()
        total_loss = 0
        total_grad_norm = 0.0
        for batch_idx, (z, c, x, X1, mu) in enumerate(dataloader_train[0]):
            z, c, x, X1, mu = [t.to(device) for t in (z, c, x, X1, mu)]
            c_hat = model(z)  # prediction ĉ
            idx = (batch_idx * dataloader_train[0].batch_size + torch.arange(z.size(0), device=device))
            mu_tilde = mu_global[idx] # select mu associated with the batch
            mu_tilde_add = [mu_global_add[i][idx] for i in range(len(mu_global_add))] # select mu associated with the batch

            if muloss:
                mu_batch_add = [mu_add[i][idx] for i in range(len(mu_add))] # select mu associated with the batch
                X1_batch_add = [X1_add[i][idx] for i in range(len(X1_add))] # select X1 associated with the batch

            # Update mu_global
            if epoch % step_mu == 0:
                optimizer_mu.optim_mu(c_batch=c_hat.detach(),main = mains[0],verbose=False, max_iter=num_iter_mu, mu_init=mu_tilde)
                mu_tilde = optimizer_mu.get_mu().detach()
                mu_global[idx] = mu_tilde

                for i in range(1,len(mains)):
                    optimizer_mu.optim_mu(c_batch=c_hat.detach(),main = mains[i],verbose=False, max_iter=num_iter_mu, mu_init=mu_tilde_add[i-1])
                    mu_tilde_add[i-1] = optimizer_mu.get_mu().detach()
                    mu_global_add[i-1][idx] = mu_tilde_add[i-1]

            # Forward and Backward pass
            if combine == "sum":
                mu_tilde_sum = mu_tilde.sum(dim=1) # Shape (batch_size, num_item)
                mu_sum = mu.sum(dim=1) if muloss else 0 # Shape (batch_size, num_item)
                loss = diff(c_hat + mu_tilde_sum, c + mu_sum, X1).to(device)

                for i in range(1,len(mains)):
                    mu_tilde_add_sum = mu_tilde_add[i-1].sum(dim=1)
                    mu_batch_add_sum = 0 if not muloss else mu_batch_add[i-1].sum(dim=1)
                    X1_ = x if not muloss else X1_batch_add[i-1]     # comme ça on a plus besoin des X1* pour la loss de Tias
                    loss += diff(c_hat + mu_tilde_add_sum, c + mu_batch_add_sum, X1_).to(device)
            
            elif combine == "random":
                k = np.random.randint(0, len(mains))
                if k == 0:
                    mu_tilde_sum = mu_tilde.sum(dim=1)
                    mu_sum = mu.sum(dim=1) if muloss else 0
                    loss = diff(c_hat + mu_tilde_sum, c + mu_sum, X1).to(device)
                else:
                    mu_tilde_add_sum = mu_tilde_add[k-1].sum(dim=1)
                    mu_batch_add_sum = 0 if not muloss else mu_batch_add[k-1].sum(dim=1)
                    X1_ = x if not muloss else X1_batch_add[k-1]
                    loss = diff(c_hat + mu_tilde_add_sum, c + mu_batch_add_sum, X1_).to(device)

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
            print(f"Epoch {epoch} | loss: {mean_loss:.4f} | train time : {train_time:.4f}", flush=True)

        ## evaling step (if needed)##
        if epoch % eval_freq == eval_freq-1:
            with torch.no_grad():
                model.eval()
                all_regrets = []

                for z, c, x, _, _ in dataloader_eval:
                    z = z.to(device)
                    c_hat = model(z)  # [batch, n]

                    # prepare the inputs for the helper
                    c_hat_np = c_hat.detach().cpu().numpy()   # shape [B,n]
                    x_true_cpu = x.float().cpu().numpy()               # shape [B,n], torch Tensor
                    c_true_cpu = c.float().cpu().numpy()  

                    for i in range(z.size(0)):
                        eval_solver.setObj(c_hat_np[i]) 
                        x_hat, _ = eval_solver.solve()
                        num = (c_true_cpu[i] * (x_true_cpu[i] - x_hat)).sum()
                        den = max((c_true_cpu[i] * x_true_cpu[i]).sum(), 1e-6)
                        rel_regret = num / den
                        all_regrets.append(rel_regret)

                mean_relat_regret = float(np.mean(all_regrets))
                std_relat_regret  = float(np.std(all_regrets))

                # Compute difference between optimal mu*(c) and adaptive mu
                mu_data = dataloader_train[0].dataset.tensors[4].to(device)
                mu_diff = torch.norm(mu_data - mu_global, p='fro').item()

                if run is not None:
                    # Log results in wandb
                    run.log({"epoch": epoch, "Mean relative regret": mean_relat_regret,
                            "Std relative regret": std_relat_regret, "norm_diff_mu": mu_diff,
                            "train_time": train_time})
                if verbose:
                    print(f"Eval Epoch {epoch} | Mean relative regret: {mean_relat_regret:.4f} | norm_diff_mu: {mu_diff:.4f}", flush=True)
                
                # Early stopping
                if mean_relat_regret < best_relat_regret - min_delta:
                    best_relat_regret = mean_relat_regret
                    epochs_no_improvement = 0
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    best_epoch = epoch
                    best_model.load_state_dict(model.state_dict())
                elif patience is not None:
                    epochs_no_improvement += 1
                    if epochs_no_improvement >= patience:
                        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}", flush=True)
                        break

        for i,t in enumerate(report_times):
            if train_time >= t and pending[t] and t > 0:
                pending[t] = False
                best_model.eval()
                regrets= test(best_model, dataloader_test, eval_solver, device, run=None)
                regrets_eval = test(best_model, dataloader_eval, eval_solver, device, run=None)
                result[i] = torch.tensor(regrets, dtype=torch.float32, device=device)
                result_eval[i] = torch.tensor(regrets_eval, dtype=torch.float32, device=device)

        # Check time limit
        if time_limit is not None:
            if train_time > time_limit:
                print("Time limit reached, stopping training.", flush=True)
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
    return result_eval,result

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


    print(f"Test relative regret: {mean_relat:.4f} ± {std_relat:.4f}", flush=True)
    if run:
        run.log({
            "test/mean_relative_regret": mean_relat,
            "test/std_relative_regret": std_relat
        })

    return rel_regr
    #return mean_relat, std_relat
