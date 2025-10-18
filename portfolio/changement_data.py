#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recalcule X et μ dans un fichier train_*.txt puis réécrit le fichier en place
(sans générer de copie .bak).

Exemple :
python update_mu_X.py \
    --fname portfolio/datasets/train_200_10000_5_8_2-25.txt \
    --iters 500 \
    --principal_lin 0          # 1 = lin→quad, 0 = quad→lin
"""

import argparse
import time
from datetime import datetime

import numpy as np
import torch

from portfolio.data_import import ImportDataset
from portfolio.my_solver import BatchSolverLin, BatchSolverQuad, BatchSolverExact
from opti_X_mu_CPU import OptimizationBatchModel

# --------------------------------------------------------------------------- #
#  Détection automatique du device (même logique que gen_data.py)
# --------------------------------------------------------------------------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#  Fonctions utilitaires
# --------------------------------------------------------------------------- #
def load_file(fname):
    with open(fname, "r") as f:
        lines = f.readlines()
    n_data, n_feat, n_item, deg, gamma = map(float, lines[0].split(","))
    meta = dict(
        num_data=int(n_data),
        num_feat=int(n_feat),
        num_item=int(n_item),
        deg=int(deg),
        gamma=float(gamma),
    )
    return meta, lines


def write_file(fname, lines):
    with open(fname, "w") as f:
        f.writelines(lines)


def recompute_X_mu(path, iters, principal_lin):
    ds = ImportDataset(path)
    n, m = ds.num_data, ds.num_item
    gamma = float(ds.gamma)
    cov = ds.get_cov(tensor=False).astype(np.float64)
    cov2 = 1e5 * cov
    c = ds.c

    lin_solver = BatchSolverLin(m, device)
    quad_solver = BatchSolverQuad(m, cov2, gamma, device)
    exact_quad_solver = BatchSolverExact(m, cov2, gamma, device)

    solvers = (
        [lin_solver, quad_solver]
        if principal_lin
        else [quad_solver, lin_solver]
    )

    optimizer = OptimizationBatchModel(solvers)
    c_tensor = torch.tensor(c, dtype=torch.float32, device=device)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    optimizer.optim_mu(c_batch=c_tensor, verbose=False, max_iter=iters)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t1 = time.perf_counter()

    X = optimizer.get_X()[:, 0, :].cpu().numpy()
    mu = optimizer.get_mu().view(n, -1).cpu().numpy()
    return X, mu, t1 - t0


def update_lines(original, meta, new_X, new_mu):
    n_feat, n_item = meta["num_feat"], meta["num_item"]
    updated = original[: 1 + n_item]  # header + covariance
    base = 1 + n_item
    for i in range(meta["num_data"]):
        parts = original[base + i].rstrip("\n").split(",")
        start_X = n_feat + 2 * n_item
        start_mu = n_feat + 3 * n_item
        parts[start_X:start_mu] = [str(v) for v in new_X[i]]
        parts[start_mu:] = [str(v) for v in new_mu[i]]
        updated.append(",".join(parts) + "\n")
    return updated


# --------------------------------------------------------------------------- #
#  Script principal
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Recalcule μ et X et met à jour un dataset train (sans .bak)."
    )
    parser.add_argument(
        "--fname", required=True, type=str, help="Chemin du fichier train_*.txt"
    )
    parser.add_argument(
        "--iters", default=500, type=int, help="Nombre d'itérations pour optim_mu"
    )
    parser.add_argument(
        "--principal_lin",
        default=0,
        type=int,
        choices=[0, 1],
        help="1 = lin→quad, 0 = quad→lin",
    )
    args = parser.parse_args()

    meta, lines = load_file(args.fname)
    print(
        f"Dataset : {args.fname} | {meta['num_data']} instances | {args.iters} itérations "
        f"{meta['num_item']} items • γ={meta['gamma']} • deg={meta['deg']}"
    )
    print(f"Device détecté : {device}")

    X, mu, elapsed = recompute_X_mu(
        args.fname, args.iters, bool(args.principal_lin)
    )
    print(
        f"Optimisation terminée en {elapsed:.2f}s "
        f"(avg {elapsed/meta['num_data']:.4f}s/instance)"
    )

    new_lines = update_lines(lines, meta, X, mu)
    write_file(args.fname, new_lines)
    print("Fichier mis à jour avec les nouveaux X et μ.")


if __name__ == "__main__":
    main()