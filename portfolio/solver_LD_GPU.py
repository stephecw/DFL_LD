import torch
from numba import cuda
import numpy as np


@cuda.jit
def dp_knapsack_gpu_batch(Lglobal_bound, capacities, Lweights, Lval, N, M, value_var_solution, Ldp):
    idx_batch = 0
    idx_thread = cuda.grid(1)
    while idx_thread >= M[idx_batch]:
        idx_thread -= M[idx_batch]
        idx_batch += 1
    idx_constraint = idx_thread
    offset_n_m = 0
    offset_m = 0
    for i in range(idx_batch):
        offset_n_m += N[i] * M[i]
        offset_m += M[i]

    capacity = capacities[offset_m + idx_constraint]
    weights = Lweights[offset_m + idx_constraint]
    val = Lval[offset_m + idx_constraint]
    for i in range(1, N[idx_batch] + 1):
        for w in range(capacity + 1):
            if weights[i - 1] <= w:
                Ldp[offset_m + idx_constraint][w][i] = max(Ldp[offset_m + idx_constraint][w][i - 1],
                                                           Ldp[offset_m + idx_constraint][w - weights[i - 1]][i - 1] + val[i - 1])
            else:
                Ldp[offset_m + idx_constraint][w][i] = Ldp[offset_m + idx_constraint][w][i - 1]

    # Backtracking to find selected items
    w = capacity
    for i in range(N[idx_batch], 0, -1):
        if Ldp[offset_m + idx_constraint][w][i] != Ldp[offset_m + idx_constraint][w][i - 1]:
            value_var_solution[offset_n_m + i - 1 + N[idx_batch] * idx_constraint] = 1
            w -= weights[i - 1]

    cuda.atomic.add(Lglobal_bound, 0, Ldp[offset_m + idx_constraint][capacity][N[idx_batch]])

def solve_knapsack_gpu_batch(problems, u):
    batch_size = len(problems)
    compteur_m = 0
    compteur_n_m = 0
    N = []
    M = []
    value_var_solution = []
    Lweights = []
    Lval = []
    capacities = []
    
    for idx_batch in range(batch_size):
        N.append(problems[idx_batch][1])  # nb_items
        M.append(problems[idx_batch][0])  # nb_constraints

    max_n = max(N)
    
    for idx_batch in range(batch_size):
        n = N[idx_batch]
        m = M[idx_batch]
        value_var_solution.extend([0] * n * m)
        Lweights.extend([0] * m)
        Lval.extend([0] * m)
        u_1 = []
        for i in range(n):
            temp = 0
            for j in range(1, m):
                temp += u[compteur_n_m + (j-1) * n + i].item()  # Convert to scalar
            u_1.append(temp)

        for idx_constraint in range(m):
            val = []
            capacity = problems[idx_batch][2 + n + idx_constraint]
            for i in range(max_n):
                if i < n:
                    if idx_constraint == 0:
                        val.append(u_1[i] + problems[idx_batch][2 + i])
                    else:
                        val.append(-u[compteur_n_m + (idx_constraint-1) * n + i].item())  # Convert to scalar
                else:
                    val.append(0)

            weights = []
            for i in range(max_n):
                if i < n:
                    weights.append(problems[idx_batch][2 + n + m + idx_constraint * n + i])
                else:
                    weights.append(0)
            Lweights[compteur_m + idx_constraint] = weights
            Lval[compteur_m + idx_constraint] = val
            capacities.append(capacity)
        compteur_n_m += N[idx_batch] * (M[idx_batch]-1)
        compteur_m += M[idx_batch]

    # Convert lists to PyTorch tensors and then to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Lglobal_bound = torch.tensor([0], dtype=torch.int32, device=device)
    value_var_solution = torch.tensor(value_var_solution, dtype=torch.int32, device=device)

    Lweights_tensor = torch.tensor(Lweights, dtype=torch.int32, device=device)
    Lval_tensor = torch.tensor(Lval, device=device)
    capacities_tensor = torch.tensor(capacities, dtype=torch.int32, device=device)
    Ldp = torch.zeros((len(capacities), torch.max(capacities_tensor) + 1, max(N) + 1), dtype=torch.int32, device=device)
    N_tensor = torch.tensor(N, dtype=torch.int32, device=device)
    M_tensor = torch.tensor(M, dtype=torch.int32, device=device)

    dp_knapsack_gpu_batch[compteur_m, 1](Lglobal_bound, 
                                                  capacities_tensor, 
                                                  Lweights_tensor, 
                                                  Lval_tensor, 
                                                  N_tensor, 
                                                  M_tensor, 
                                                  value_var_solution, Ldp)
    return value_var_solution