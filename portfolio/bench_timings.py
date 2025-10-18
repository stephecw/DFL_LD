#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark timings for:
1) Exact solves to compute x_star for each instance.
2) Optimization of mu and X via optimizer.optim_mu.

Usage example:
python bench_timings.py --fname portfolio/datasets/train_200_10000_5_8_2-25.txt --iters 500 --principal_lin 0 --csv timings_results.csv
"""

import argparse
import csv
import os
import time
from datetime import datetime

import numpy as np
import torch

from portfolio.data_import import ImportDataset
from pyepo.model.grb import portfolioModel
from portfolio.my_solver import BatchSolverLin, BatchSolverQuad, BatchSolverExact
from opti_X_mu_CPU import OptimizationBatchModel

def main():
    parser = argparse.ArgumentParser(description="Bench x* loop and optim_mu timings on an existing train dataset.")
    parser.add_argument("--fname", type=str, required=True, help="Path to existing train_*.txt dataset.")
    parser.add_argument("--iters", type=int, default=500, help="max_iter for optimizer.optim_mu.")
    parser.add_argument("--principal_lin", type=int, default=0, help="1: main subproblem is linear first, 0: quadratic first (as in your default).")
    parser.add_argument("--csv", type=str, default="timings_results.csv", help="CSV file to append results.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device for batch solvers.")
    parser.add_argument("--seed", type=int, default=135, help="Seed (only used if you add stochastic parts).")
    args = parser.parse_args()

    # Load dataset
    ds = ImportDataset(args.fname, model=None, z_stats=None)
    num_data = ds.num_data
    num_item = ds.num_item
    deg = int(ds.deg)
    gamma = float(ds.gamma)
    cov = ds.get_cov(tensor=False).astype(np.float64)  # keep as numpy for GRB wrapper
    c = ds.c  # shape (num_data, num_item)

    # Choose device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # --- Phase 1: exact solves for x_star ---
    x_star_list = []
    t0 = time.perf_counter()
    for i in range(num_data):
        model = portfolioModel(num_assets=num_item, covariance=cov, gamma=gamma)
        model.setObj(c[i])
        x_star, _ = model.solve()
        x_star_list.append(x_star)
    t1 = time.perf_counter()
    t_xstar = t1 - t0
    avg_xstar = t_xstar / num_data if num_data > 0 else float("nan")

    # --- Phase 2: optimization of mu (and X) ---
    cov2 = 1e5 * cov  # as in your generation script
    c_tensor = torch.tensor(c, dtype=torch.float32)

    lin_solver = BatchSolverLin(num_item, device)
    quad_solver = BatchSolverQuad(num_item, cov2, gamma, device)
    exact_quad_solver = BatchSolverExact(num_item, cov2, gamma, device)  # constructed for parity, not used directly

    if args.principal_lin == 1:
        solvers = [lin_solver, quad_solver]
        solver_order = "lin->quad"
    else:
        solvers = [quad_solver, lin_solver]
        solver_order = "quad->lin"

    optimizer = OptimizationBatchModel(solvers)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t2 = time.perf_counter()
    optimizer.optim_mu(c_batch=c_tensor[0:num_data], verbose=False, max_iter=args.iters)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t3 = time.perf_counter()
    t_optim = t3 - t2
    avg_optim = t_optim / num_data if num_data > 0 else float("nan")

    # Prepare CSV row(s)
    now_iso = datetime.now().isoformat(timespec="seconds")
    rows = [
        {
            "timestamp": now_iso,
            "phase": "x_star_loop",
            "total_seconds": f"{t_xstar:.6f}",
            "avg_seconds_per_instance": f"{avg_xstar:.9f}",
            "num_data": num_data,
            "num_item": num_item,
            "deg": deg,
            "gamma": gamma,
            "iters": args.iters,
            "principal_lin": args.principal_lin,
            "solver_order": solver_order,
            "device": str(device),
            "dataset_path": args.fname,
        },
        {
            "timestamp": now_iso,
            "phase": "optim_mu",
            "total_seconds": f"{t_optim:.6f}",
            "avg_seconds_per_instance": f"{avg_optim:.9f}",
            "num_data": num_data,
            "num_item": num_item,
            "deg": deg,
            "gamma": gamma,
            "iters": args.iters,
            "principal_lin": args.principal_lin,
            "solver_order": solver_order,
            "device": str(device),
            "dataset_path": args.fname,
        },
    ]

    # Write / append CSV
    fieldnames = list(rows[0].keys())
    write_header = not os.path.exists(args.csv)

    with open(args.csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print("=== Results ===")
    print(f"x* loop:   {t_xstar:.3f}s total  | {avg_xstar*1000:.3f} ms/instance")
    print(f"optim_mu:  {t_optim:.3f}s total  | {avg_optim*1000:.3f} ms/instance")
    print(f"Appended to: {args.csv}")
    print(f"Order: {solver_order} | Device: {device}")

if __name__ == "__main__":
    main()
