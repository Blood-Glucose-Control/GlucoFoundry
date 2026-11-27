# cgmacros_ppgr/configs/

Experiment-owned configuration. The toolkit ships generic schemas and baseline
templates; this directory holds **this experiment's** tuned values.

- `data/` — dataset/holdout configs for `cgmacros_ppgr` (seeded from the
  `patient_level` holdout strategy, held-out fraction `0.2`).
- `sweeps/` — Optuna sweep configs (study specs) for this experiment.

Versioned configs always stay in the repo; large run artifacts
(`../results/`, `../trained_models/`) are gitignored and can be redirected to
scratch/external storage via `GLUCOFOUNDRY_ARTIFACTS_ROOT`.
