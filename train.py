import time
import torch
import numpy as np
from gurobipy import GRB
import os
import csv
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train(model, diff_list, eval_solver, dataloader_train, dataloader_eval, dataloader_test, optimizer, scheduler,
          checkpoints, num_eval_per_cp=5, output_file=None,
          approach="classic", loss_with_mu=False,
          decompositions=None, freq_dec_change=1,
          step_mu=None, num_iter_mu=None, optimizer_mu=None,
          mu_global0=None,
          run=None, verbose=False, param=None, metric = "time", device=device):
    """
    Training a DFL-model by minimizing LD loss, with adaptive mu.

    Args:
        model: ML model to train
        diff_method: Differentiation technique used to compute loss gradient
        eval_solver: solver for the problem, used during evaluation
        dataloader_train: DataLoader for training (z, c, x, X1*(c), mu(c))
        dataloader_eval: DataLoader for eval (z, c, x, X1*(c), mu(c))
        optimizer: PyTorch optimizer for training
        scheduler: PyTorch scheduler
        epochs: max number of training epochs
        checkpoints: list of time checkpoints to save the best model
        eval_freq: frequency of evaling (in time)
        decompositions: list of decomposition methods to use
        step_mu: frequency of updating mu (in epochs)
        num_iter_mu: number of sub-gradient descent steps when updating mu
        optimizer_mu: optimizer object to update mu
        approach: 'LD' for Lagrangian Decomposition, 'SG' for Sub-gradient descent, None for classic DFL and MSE
        loss: O for loss with penalty term, 1 for loss without penalty term
        mu_global0: initial value of mu_global shape (len(decompositions), batch_size, dim-1, num_items)
        run: wandb.run for logging results
        verbose: bool: If True, print training info
    """
    start_time = time.time()
    
    idx_checkpoint = 0  # Index for the current checkpoint
    save = []
    
    # Initialize variables for best model tracking
    best_relat_regret = float("inf")
    best_relat_regret_std = 0
    best_model_state = None
    best_epoch = 0
    time_best_epoch = 0.0
    
    mu_global = mu_global0
    main_pb = decompositions[0]  
    
    train_time = 0
    last_eval_time = 0.
    if metric=="time":
        eval_limit = checkpoints[idx_checkpoint] / num_eval_per_cp
    else:
        eval_limit = checkpoints[idx_checkpoint] // num_eval_per_cp 
        
    epoch = -1  # Initialize epoch counter
    while(True):
        epoch += 1 
        previous_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}        
        epoch_start_time = time.time()

        ## Training step ##
        model.train()
        total_loss = 0
        
        # Randomly select a decomposition method
        if len(decompositions) != 1 and (approach == "SG" or approach == "LD") and epoch % freq_dec_change == 0:
            main_pb = decompositions[np.random.randint(len(decompositions))]
            
        for batch_idx, (z, c, x, X1, mu) in enumerate(dataloader_train):
            c_hat = model(z)  # prediction ĉ
            mu_tilde_sum = 0
            mu_sum = 0
            x_ = x
            if approach == 'LD' or approach == 'SG':
                x_ = X1[:, main_pb]  # Use X1 for LD and SG approaches
                if loss_with_mu:
                    mu_sum = mu[:, main_pb].sum(dim=1)  # Shape (batch_size, num_item)         
            if approach == 'LD':
                mu_tilde_sum = mu[:, main_pb].sum(dim=1)  # Shape (batch_size, num_item)     
            elif approach == 'SG':
                idx = (batch_idx * dataloader_train.batch_size + torch.arange(z.size(0), device=device))
                if epoch % step_mu == 0:
                    for idx_pb in decompositions:
                        mu_tilde = mu_global[idx,idx_pb]
                        optimizer_mu.optim_mu(c_batch=c_hat.detach(), main_solver=idx_pb, verbose=False, mu_init=mu_tilde, max_iter=num_iter_mu)
                        mu_tilde = optimizer_mu.get_mu(tensor=True, device=device)
                        mu_global[idx,idx_pb] = mu_tilde
                mu_tilde_sum = mu_global[idx,main_pb].sum(dim=1)

            # Forward pass
            loss = diff_list[main_pb](c_hat + mu_tilde_sum, c + mu_sum, x_)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        mean_loss = total_loss / len(dataloader_train)
        # Update the learning rate scheduler if applicable
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(mean_loss)
            else:
                scheduler.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        train_time += epoch_duration
        
        if verbose and (epoch%1000==0):
            print(f"Epochs done {epoch+1} : mean_loss {mean_loss}, main_pb {main_pb}")
        if run is not None:
            current_lr = get_learning_rate(optimizer)
            mu_data = dataloader_train.dataset.tensors[4].to(device)
            mu_diff = torch.norm(mu_data - mu_global, p='fro').item()
            # Log results in wandb
            run.log({"num_epochs": epoch+1, "train_loss": mean_loss, "epoch_duration": epoch_duration,
                    "train_time": train_time, "lr": current_lr, 'mu_diff':{mu_diff}})
            
        metric_val = train_time if metric == "time" else epoch + 1
        
        if metric_val > checkpoints[idx_checkpoint]:
            if verbose:
                print(f"Checkpoint {idx_checkpoint} reached ! cp : {checkpoints[idx_checkpoint]} -> train_time : {train_time}.", flush=True)
                print(f"Evaluating last epoch {epoch} model...")
            
            model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(previous_model_state)
            mean_relat_reg, std_relat_reg = model.evaluate(eval_solver, dataloader_eval)
            if mean_relat_reg < best_relat_regret:
                if verbose:
                    print(f"New best state : {mean_relat_reg} after {epoch} epochs")
                best_relat_regret = mean_relat_reg
                best_relat_regret_std = std_relat_reg
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                time_best_epoch = train_time
            model.load_state_dict(model_state)
            
            
            while idx_checkpoint < len(checkpoints) and metric_val > checkpoints[idx_checkpoint]:
                print(f"Checkpoint {idx_checkpoint} reached ! {epoch} done, saving model so far.", flush=True)
                save = {'cp':checkpoints[idx_checkpoint], 
                            'train_time':train_time,
                            'epoch':epoch, 
                            'best_epoch':best_epoch, 
                            'time_best_epoch':time_best_epoch,
                            'best_relat_regret':best_relat_regret, 
                            'best_relat_regret_std':best_relat_regret_std, 
                            'best_model_state':best_model_state
                            }
                model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(best_model_state)
                saving(model, save, param, dataloader_test, eval_solver, output_file, verbose)
                model.load_state_dict(model_state)
                idx_checkpoint += 1
            
            if idx_checkpoint >= len(checkpoints):
                print("All checkpoints reached, stopping training.", flush=True)
                break
            if metric == "time":
                eval_limit = checkpoints[idx_checkpoint] / num_eval_per_cp
            else:
                eval_limit = checkpoints[idx_checkpoint] // num_eval_per_cp
            last_eval_time = train_time if metric == "time" else epoch
        
        if (train_time - last_eval_time >= eval_limit and metric == "time") or (epoch - last_eval_time >= eval_limit and metric != "time"):
            last_eval_time = train_time if metric == "time" else epoch
            mean_relat_reg, std_relat_reg = model.evaluate(eval_solver, dataloader_eval)
            if mean_relat_reg < best_relat_regret:
                if verbose:
                    print(f"New best state : {mean_relat_reg} after {epoch+1} epochs")
                best_relat_regret = mean_relat_reg
                best_relat_regret_std = std_relat_reg
                best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch+1
                time_best_epoch = train_time


    if run is not None:
        total_duration = time.time() - start_time
        run.log({
            "total_duration": total_duration
        })
    return save

def saving(model, save, param, test_loader, eval_solver, output_file, verbose):
    # Build the model filename
    param = {'cp': save['cp'], **param}
    # model_filename = "knapsack/models/lin"
    # for v in param.values():
    #     model_filename += f"_{v}" if v != '' else ''
    # model_filename += ".pth"
    
    # if verbose:
    #     print(f"Saving model to {model_filename}", flush=True)
    # torch.save(model.state_dict(), model_filename)
    
    # Test the model
    if verbose:
        print(f"Testing model.", flush=True)
    regrets_test = test(model, test_loader, eval_solver)
    mean_relat_test = np.mean(regrets_test)
    median_relat_test = np.median(regrets_test)
    std_relat_test = np.std(regrets_test)
    
    # Build the output row
    row = {
        'cp': save['cp'], **param,
        'train_time': save['train_time'], 
        'epoch': save['epoch'],
        'best_epoch': save['best_epoch'],
        'time_best_epoch': save['time_best_epoch'],
        'mean_relat_eval': save['best_relat_regret'],
        'std_relat_eval': save['best_relat_regret_std'],
        'mean_relat_test': mean_relat_test,
        'median_relat_test': median_relat_test,
        'std_relat_test': std_relat_test,
        'regrets_test': regrets_test
    }

    write_header = not os.path.exists(output_file)
    
    # Write to CSV
    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['cp', 'train_time',
                      'epoch', 'best_epoch', 'time_best_epoch', 'mean_relat_eval', 'std_relat_eval',
              'mean_relat_test', 'median_relat_test', 'std_relat_test'] + list(param.keys()) + ['regrets_test']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if write_header:
            writer.writeheader()
        
        # Serialize regrets_test as a string to preserve the CSV structure
        row['regrets_test'] = ','.join(map(str, row['regrets_test']))
        writer.writerow(row)

def test(model, test_loader, eval_solver):
    """
    Compute relative regret on a test set.

    Args:
        model       : trained nn.Module
        test_loader : DataLoader yielding (z, c, x, *_)
        eval_solver : solver with setObj()/solve() interface

    Returns:
        relative regret
    """
    model.eval()
    rel_regr = []
    with torch.no_grad():
        for z, c, x, *_ in test_loader:
            c_hat = model(z)
            for i in range(z.size(0)):
                eval_solver.setObj(c_hat[i])
                x_pred_np, _ = eval_solver.solve()

                x_true = x[i].float()
                c_true = c[i]
                x_pred = torch.Tensor(x_pred_np)

                num = torch.dot(c[i], x[i] - x_pred)
                if eval_solver.modelSense == GRB.MINIMIZE:
                    num = -num
                den = torch.dot(c_true, x_true).clamp(min=1e-6)
                rel_regr.append((num / den).item())
    return rel_regr
