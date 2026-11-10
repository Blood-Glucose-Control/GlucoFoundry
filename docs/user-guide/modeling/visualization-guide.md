# Visualization Guide and Example Gallery

This page documents the maintained visualization scripts in [experiments/nocturnal/scripts/visualization/](/experiments/nocturnal/scripts/visualization), their duplication assessment status, and a clean example artifact for each script.

## Example storage location

For MkDocs-facing visualization examples, the canonical committed location is:

- [docs/assets/visualizations/examples/](/docs/assets/visualizations/examples)

For scripts that naturally produce multiple files (appendix panels or per-episode outputs), examples live in script-specific subfolders under that directory.

## Duplication assessment (Wave 4)

| Family | Scripts | Duplication level | Decision |
|---|---|---:|---|
| Episode overlays | `plot_forecast_episode_overlays.py`, `plot_forecast_comparison.py` | Low | Keep separate (single-model uncertainty overlays vs multi-model head-to-head episode comparison). |
| RMSE vs horizon | `plot_rmse_vs_horizon.py`, `plot_rmse_vs_horizon_grid.py` | Medium | Keep both (distributional boxplots vs cross-model comparative grid); shared loading/stats already centralized in [src/visualization/nocturnal.py](/src/visualization/nocturnal.py). |
| PIT / calibration | `plot_pit_histograms.py`, `plot_pit_horizon_heatmap.py`, `plot_pit_combined.py`, `plot_reliability_diagrams.py` | Medium | Keep all four (complementary calibration diagnostics at global, per-horizon, and quantile-reliability levels). |
| Coverage/sharpness | `plot_coverage_error_2d.py`, `plot_coverage_error_2d_sharpness.py`, `plot_coverage_sharpness_scatter.py` | Medium | Keep all three (different 2D/3D encodings and readability trade-offs for calibration-vs-sharpness). |
| Percentile grid | `plot_probabilistic_forecast_grid.py` | Low | Keep (dataset-by-percentile interpretability surface not duplicated elsewhere). |

## Per-script examples and interpretation

### `plot_forecast_comparison.py`

**Stored example:** [plot_forecast_comparison.png](/docs/assets/visualizations/examples/plot_forecast_comparison.png)

![plot_forecast_comparison](../../assets/visualizations/examples/plot_forecast_comparison.png)

Use this to compare competing models on identical episodes. Look for consistent wins on hard-percentile columns and threshold-tracking behavior near hypo regions.

### `plot_coverage_error_2d.py`

**Stored example:** [plot_coverage_error_2d.png](/docs/assets/visualizations/examples/plot_coverage_error_2d.png)

![plot_coverage_error_2d](../../assets/visualizations/examples/plot_coverage_error_2d.png)

Use this to assess 50% and 80% interval calibration jointly. Best models cluster near the origin (0,0).

### `plot_coverage_error_2d_sharpness.py`

**Stored example:** [plot_coverage_error_2d_sharpness.png](/docs/assets/visualizations/examples/plot_coverage_error_2d_sharpness.png)

![plot_coverage_error_2d_sharpness](../../assets/visualizations/examples/plot_coverage_error_2d_sharpness.png)

Use this when you need calibration error and sharpness in the same visual. Look for points near origin with smaller/lower-sharpness encodings.

### `plot_coverage_sharpness_scatter.py`

**Stored example:** [plot_coverage_sharpness_scatter.png](/docs/assets/visualizations/examples/plot_coverage_sharpness_scatter.png)

![plot_coverage_sharpness_scatter](../../assets/visualizations/examples/plot_coverage_sharpness_scatter.png)

Use this to inspect calibration-vs-sharpness tradeoffs by PI level. Bottom-center clusters indicate good calibration with tighter intervals.

### `plot_forecast_episode_overlays.py`

**Stored examples:**
- [combined.png](/docs/assets/visualizations/examples/plot_forecast_episode_overlays/combined.png)
- [plot_forecast_episode_overlays/](/docs/assets/visualizations/examples/plot_forecast_episode_overlays)

![plot_forecast_episode_overlays](../../assets/visualizations/examples/plot_forecast_episode_overlays/combined.png)

Use this for episode-level qualitative inspection of mean and interval behavior. Look for interval escapes and forecast lag around rapid BG transitions.

### `plot_pit_combined.py`

**Stored example:** [pit_combined.png](/docs/assets/visualizations/examples/plot_pit_combined/pit_combined.png)

![plot_pit_combined](../../assets/visualizations/examples/plot_pit_combined/pit_combined.png)

Use this as the compact calibration dashboard: histogram for global shape errors and heatmap for horizon-localized drift.

### `plot_pit_histograms.py`

**Stored examples:**
- [pit_main_body.png](/docs/assets/visualizations/examples/plot_pit_histograms/pit_main_body.png)
- [plot_pit_histograms/](/docs/assets/visualizations/examples/plot_pit_histograms)

![plot_pit_histograms](../../assets/visualizations/examples/plot_pit_histograms/pit_main_body.png)

Use this for aggregate calibration shape checks (uniform vs U-shape/skew/hump patterns).

### `plot_pit_horizon_heatmap.py`

**Stored examples:**
- [pit_horizon_heatmap.png](/docs/assets/visualizations/examples/plot_pit_horizon_heatmap/pit_horizon_heatmap.png)
- [pit_mad_over_horizon.png](/docs/assets/visualizations/examples/plot_pit_horizon_heatmap/pit_mad_over_horizon.png)

![plot_pit_horizon_heatmap](../../assets/visualizations/examples/plot_pit_horizon_heatmap/pit_horizon_heatmap.png)

Use this to locate where in the horizon calibration fails. Look for strong color shifts toward later bins.

### `plot_probabilistic_forecast_grid.py`

**Stored example:** [plot_probabilistic_forecast_grid.png](/docs/assets/visualizations/examples/plot_probabilistic_forecast_grid.png)

![plot_probabilistic_forecast_grid](../../assets/visualizations/examples/plot_probabilistic_forecast_grid.png)

Use this for percentile-stratified episode behavior by dataset. Compare interval adequacy and median-shape degradation from easy to hard columns.

### `plot_reliability_diagrams.py`

**Stored examples:**
- [reliability_main_body.png](/docs/assets/visualizations/examples/plot_reliability_diagrams/reliability_main_body.png)
- [plot_reliability_diagrams/](/docs/assets/visualizations/examples/plot_reliability_diagrams)

![plot_reliability_diagrams](../../assets/visualizations/examples/plot_reliability_diagrams/reliability_main_body.png)

Use this to evaluate quantile calibration directly against the diagonal; larger ECE and off-diagonal curves indicate worse calibration.

### `plot_rmse_vs_horizon.py`

**Stored example:** [plot_rmse_vs_horizon.svg](/docs/assets/visualizations/examples/plot_rmse_vs_horizon.svg)

![plot_rmse_vs_horizon](../../assets/visualizations/examples/plot_rmse_vs_horizon.svg)

Use this for per-horizon error distribution, not just average error. Watch for widening boxes and high whiskers at long horizons.

### `plot_rmse_vs_horizon_grid.py`

**Stored example:** [plot_rmse_vs_horizon_grid.svg](/docs/assets/visualizations/examples/plot_rmse_vs_horizon_grid.svg)

![plot_rmse_vs_horizon_grid](../../assets/visualizations/examples/plot_rmse_vs_horizon_grid.svg)

Use this to compare model rank trajectories across horizon and dataset. Focus on crossings and IQR inflation.
