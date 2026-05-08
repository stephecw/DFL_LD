# Quadratic portfolio optimization

This directory contains the data-generation and training pipeline for the
quadratic portfolio optimization benchmark.

> All commands below should be executed from the **repository root**.

## 1. Data generation

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

## 2. Training

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
