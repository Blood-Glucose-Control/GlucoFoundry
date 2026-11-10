# Statistical Baseline Configs

Three statistical baselines via AutoGluon. A training step is required
(AutoGluon fits to compute residuals for quantile synthesis).

| Config | Model | Datasets | Notes |
|--------|-------|---------|-------|
| `00_autoarima_bg_only.yaml` | AutoARIMA | all 4 | seasonal=False |
| `01_autoarima_bg_iob.yaml` | AutoARIMA + IOB | aleppo, brown, lynch | seasonal=False |
| `02_theta_bg_only.yaml` | Theta | all 4 | season_length=1 |
| `03_npts_bg_only.yaml` | NPTS | all 4 | natively probabilistic |

All configs apply a **2-hour `time_limit`** cap. Series not fit in time
fall back to Naive — no work is lost.

## Run

The earlier hand-rolled per-model sweep shells were removed. These CPU
statistical configs run through the generic forecasting workflow in
`scripts/workflows/forecasting/`.
