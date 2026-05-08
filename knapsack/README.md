# Multi-dimensional knapsack

This directory contains the data-generation and training pipeline for the
multi-dimensional knapsack benchmark.

> All commands below should be executed from the **repository root**.

## 1. Data generation

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

This produces four files in `knapsack/datasets/`:

- `train_base_{dim}_{n_feat}_{n}_{n_train}_{deg}.txt` — intermediate, no LD vars
- `eval_{dim}_{n_feat}_{n}_{n_eval}_{deg}.txt`
- `test_{dim}_{n_feat}_{n}_{n_test}_{deg}.txt`
- `train_{dim}_{keep}_{main}_{n_feat}_{n}_{n_train}_{deg}.txt` — with `X*`, `mu*`

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

## 2. Training

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
