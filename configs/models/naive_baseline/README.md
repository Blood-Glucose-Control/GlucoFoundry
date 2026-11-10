# Naive Baseline Configs

Two zero-parameter baselines. AutoGluon fits Naive/Average in seconds.

| Config | Model | Datasets |
|--------|-------|---------|
| `00_naive.yaml` | Naive (last-value carry-forward) | all 4 |
| `01_average.yaml` | Average (historical mean) | all 4 |

> **Note:** PIT histograms will show miscalibration. This is intentional —
> synthetic quantiles from residuals contrast with natively probabilistic
> models (NPTS, DeepAR, PatchTST, TFT, Chronos-2) in the paper.

## Run

The earlier hand-rolled per-model sweep shells were removed. These AutoGluon
baseline configs run through the generic forecasting workflow in
`scripts/workflows/forecasting/`.
