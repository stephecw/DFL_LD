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
[requirements.txt](requirements.txt).

A minimal install via conda:

```bash
conda create -n dfl_ld python=3.12
conda activate dfl_ld
pip install -r requirements.txt
```

All commands below should be executed from the **repository root**.

## Workflow

The typical workflow is the same for both benchmarks:

1. **Generate datasets** with `gen_data.py` (one per problem). The script
   produces `(features, costs, primal optimum)` for the base dataset, then
   appends the LD-related quantities `(X*_1, mu*)` required by the LD/SG
   approaches.
2. **Train** with `run_experiments.py` (one per problem). The script trains a
   model with one or more of the four approaches and writes a CSV log
   (mean / median / std test regret per checkpoint).

Per-benchmark instructions, full CLI reference and example commands:

- [`knapsack/README.md`](knapsack/README.md) — multi-dimensional knapsack
- [`portfolio/README.md`](portfolio/README.md) — quadratic portfolio optimization

## License

Released for academic and research use. Please open an issue or contact the
authors for any other use case.

## Contact

For questions about the code, open a GitHub issue or contact:

- Stéphane Eilles-Chan Way (`stephane.eilles-chan-way@polytechnique.edu`)
- Hugo Percot (`hugo.percot@polytechnique.edu`)
