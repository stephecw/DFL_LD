# Scaling Decision-Focused Learning to Large Problems with Lagrangian Decomposition

This repository contains the official implementation of the paper
*"Scaling Decision-Focused Learning to Large Problems with Lagrangian Decomposition"*
accepted at **IJCAI 2026**.

**Authors:** Stéphane Eilles-Chan Way, Hugo Percot, Quentin Cappart, Tias Guns, Louis-Martin Rousseau.

## Overview

Decision-Focused Learning (DFL) trains a predictive model whose predictions are
fed to a downstream constrained optimization problem (COP), so that the
*decisions* (rather than the predictions in isolation) are as good as possible.
DFL excels in under-specified settings but is computationally expensive: every
training step requires solving a COP for every instance.

This repository implements a framework that integrates **Lagrangian
Decomposition (LD)** into the DFL pipeline. The key idea is to replace the
expensive primal solve by a faster decomposed problem whose subproblems are
mono-constrained, while keeping the combinatorial nature of the original
problem. We propose two new loss functions (`L1` with the multiplier penalty
term, `L2` without), and combine them with two standard differentiation
techniques: `SPO+` and `IMLE`.

The framework is evaluated on two benchmarks:

- **Multi-dimensional knapsack problem** (up to 300 items, 10 constraints).
- **Quadratic portfolio optimization** (up to 400 assets).

## Repository structure

```
.
├── train.py                     # Unified training loop (classic / LD / SG / MSE)
├── opti_X_mu.py                 # GPU-accelerated batch mu-optimizer (Adam)
├── opti_X_mu_CPU.py             # CPU batch mu-optimizer (parallel + serial)
├── diff_methods.py              # SPO+, IMLE, Exact, MSE wrappers around pyepo
├── models_class.py              # Predictive model (CustomMLP)
├── utils.py                     # Seeding helpers
│
├── knapsack/
│   ├── gen_data.py              # Dataset generation (base + LD variables)
│   ├── run_experiments.py       # Training entry point for the knapsack
│   ├── data_import.py           # Dataset reader
│   ├── solver.py                # Knapsack solver wrappers (multi-D + 1D-GPU)
│   └── datasets/                # Generated training/evaluation/test files
│
└── portfolio/
    ├── gen_data.py              # Dataset generation (base + LD variables)
    ├── run_experiments.py       # Training entry point for the portfolio
    ├── data_import.py           # Dataset reader
    ├── my_solver.py             # Linear / Quadratic / Exact portfolio solvers
    ├── bench_timings.py         # Benchmarks for x* and mu solving times
    └── changement_data.py       # Recompute X, mu in an existing dataset
```

The four training approaches available in `train.py` are:

| Approach   | Description                                                                |
|------------|----------------------------------------------------------------------------|
| `classic`  | Standard DFL: differentiable solver applied to the full COP.               |
| `LD`       | Our LD-based DFL with **precomputed**, fixed Lagrangian multipliers.       |
| `SG`       | LD-based DFL with **online** Lagrangian-multiplier updates (sub-gradient). |
| `MSE`      | Mean-squared-error baseline (prediction-focused).                          |

## Installation

The code targets Python 3.12+. Required packages are listed in
[requirements.txt](requirements.txt):

- `torch` (>= 2.0)
- `numpy`, `joblib`
- `gurobipy` (a valid Gurobi license is required)
- `pyepo` (https://github.com/khalil-research/PyEPO)
- `numba` *(optional, only for the GPU 1D-knapsack solver)*
- `wandb` *(optional, only for offline experiment tracking)*
- `matplotlib` *(optional, only for the LD-gap visualization script)*

A minimal install via conda:

```bash
conda create -n dfl_ld python=3.12
conda activate dfl_ld
pip install -r requirements.txt
```

All commands below should be executed from the **repository root**.

## Workflow

The typical workflow is the following:

1. **Generate datasets** with `gen_data.py` (one per problem). The script
   produces `(features, costs, primal optimum)` for the base dataset, then
   appends the LD-related quantities `(X*_1, mu*)` required by the LD/SG
   approaches.
2. **Train** with `run_experiments.py` (one per problem). The script trains a
   model with one or more of the four approaches and writes a CSV log
   (mean / median / std test regret per checkpoint).
3. *(optional)* **Re-evaluate** a saved checkpoint or recompute LD variables on
   an existing dataset.

### 1. Multi-dimensional knapsack

#### 1.1 Data generation

Generate a base dataset and append the LD variables for one decomposition
(single-decomposition mode, `--keep K`):

```bash
python -m knapsack.gen_data \
    --n 100 --dim 10 --deg 4 \
    --n_train 200 --n_eval 100 --n_test 1000 \
    --n_feat 12 --noise 0.5 \
    --keep 1 --main 0 \
    --n_iter 10000 --conv 1e-4
```

This produces three files in `knapsack/datasets/`:

- `train_base_{dim}_{n_feat}_{n}_{n_train}_{deg}.txt` (intermediate, no LD vars)
- `eval_{dim}_{n_feat}_{n}_{n_eval}_{deg}.txt`
- `test_{dim}_{n_feat}_{n}_{n_test}_{deg}.txt`
- `train_{dim}_{keep}_{main}_{n_feat}_{n}_{n_train}_{deg}.txt` (with `X*`, `mu*`)

For multi-decomposition (`--keep -1`) the script computes `X*`, `mu*` for every
constraint and produces a single file
`train_{dim}_-1_{n_feat}_{n}_{n_train}_{deg}.txt` that stores all decompositions:

```bash
python -m knapsack.gen_data \
    --n 100 --dim 10 --deg 4 \
    --n_train 200 --n_eval 100 --n_test 1000 \
    --keep -1 --n_iter 10000 --conv 1e-4
```

Useful flags:

| Flag                          | Meaning                                                       |
|-------------------------------|---------------------------------------------------------------|
| `--n N1 N2 ...`               | List of item counts to iterate over.                          |
| `--dim D1 D2 ...`             | List of constraint counts to iterate over.                    |
| `--n_iter`                    | Maximum sub-gradient iterations to optimize each `mu*`.       |
| `--conv`                      | Convergence tolerance for the mu-optimizer (Adam).            |
| `--monitor`                   | Save the LD relative gap to disk (one file per decomposition).|

#### 1.2 Training

Run all four training approaches in a single command:

```bash
python -m knapsack.run_experiments \
    --diff SPOPlus \
    --dim 10 --n 100 --n_feat 12 --deg 4 \
    --keep 1 --mains 0 \
    --ep_classic 1 --ep_ld 1 --ep_sg 1 --ep_mse 1 \
    --report 60 300 600 \
    --lr 1e-3 \
    --step_mu 10 --n_iter_mu 30 \
    --muloss 1 \
    --save_model 1 \
    --out_file knapsack/results.csv
```

For multi-decomposition, replace the decomposition flags by `--keep -1` and
pass the indices of the decompositions you want to train on via `--mains`:

```bash
python -m knapsack.run_experiments \
    --diff IMLE --dim 10 --n 100 --deg 4 \
    --keep -1 --mains 0 1 2 3 \
    --ep_ld 1 --ep_sg 1 \
    --report 60 300 \
    --lambd 10 --sigma 1.0 --n_samples 1 --kappa 5 \
    --out_file knapsack/results.csv
```

Set `--ep_*` to `0` for the approaches you want to skip.

Important flags:

| Flag                            | Meaning                                                   |
|---------------------------------|-----------------------------------------------------------|
| `--diff {SPOPlus, IMLE}`        | Differentiation technique.                                |
| `--keep K \| -1`                | Single-decomposition (`K >= 1`) or multi-decomposition.   |
| `--mains m0 m1 ...`             | Indices of the decompositions used in training.           |
| `--report S1 S2 ...`            | Checkpoint times in seconds at which to evaluate.         |
| `--ep_{classic,ld,sg,mse}`      | Number of epochs per approach (0 disables it).            |
| `--muloss {0, 1}`               | 1 = use loss `L1` (with mu term); 0 = use loss `L2`.      |
| `--step_mu`, `--n_iter_mu`      | Online mu-update frequency and inner iterations (SG only).|
| `--save_model {0, 1}`           | Save the best checkpoint to `knapsack/models/...pth`.     |
| `--regenerate 1`                | Regenerate the datasets before training (calls gen_data). |

The CSV produced by `--out_file` contains one row per checkpoint and per
training approach with columns `cp`, `train_time`, `epoch`, `best_epoch`,
`mean_relat_eval`, `mean_relat_test`, `median_relat_test`, `std_relat_test`,
`regrets_test`, plus the run hyperparameters (`dim`, `keep`, `mains`, `deg`,
`jobtype`, `method`, `lr`, `muloss`, `step_mu`, `num_iter_mu`).

### 2. Quadratic portfolio optimization

#### 2.1 Data generation

```bash
python -m portfolio.gen_data \
    --n 50 --gamma 2.25 \
    --n_train 100 --n_validation 25 --n_test 10000 \
    --n_feat 5 --deg 8 \
    --lin 0 \
    --n_iter 500
```

This generates three files in `portfolio/datasets/`:

- `train_{n}_{n_train}_{n_feat}_{deg}_{gamma}.txt` (with `X*`, `mu*`)
- `validation_{n}_{n_validation}_{n_feat}_{deg}_{gamma}.txt`
- `test_{n}_{n_test}_{n_feat}_{deg}_{gamma}.txt`

`--lin 0` keeps the quadratic constraint as the *main* subproblem (recommended);
`--lin 1` keeps the linear constraint instead.

#### 2.2 Training

```bash
python -m portfolio.run_experiments \
    --n 50 --deg 8 \
    --method SPOPlus \
    --ep_classic 1 --ep_ld 1 --ep_sg 1 --ep_mse 1 \
    --report 60 300 600 \
    --lr 2e-3 \
    --step_mu 10 --n_iter_mu 30 \
    --muloss 1 \
    --regenerate 0 \
    --out_file portfolio/results.csv
```

Setting `--regenerate 1` will call `gen_data` automatically before training.

| Flag                              | Meaning                                                     |
|-----------------------------------|-------------------------------------------------------------|
| `--method {SPOPlus, IMLE, Exact}` | Differentiation technique. `Exact` uses the closed-form solver of the QP subproblem. |
| `--lambda_imle`, `--sigma`, `--n_samples` | IMLE hyperparameters.                              |
| `--ep_{classic,ld,sg,mse}`        | Number of epochs per approach (0 disables it).              |
| `--scheduler`                     | LR scheduler (`ReduceLROnPlateau`, `StepLR`, `OneCycleLR`, `None`). |
| `--num_eval_per_cp`               | Number of intermediate evaluations per checkpoint.          |

## License

Released for academic and research use. Please open an issue or contact the
authors for any other use case.

## Contact

For questions about the code, open a GitHub issue or contact:

- Stéphane Eilles-Chan Way (`stephane.eilles-chan-way@polytechnique.edu`)
- Hugo Percot (`hugo.percot@polytechnique.edu`)
