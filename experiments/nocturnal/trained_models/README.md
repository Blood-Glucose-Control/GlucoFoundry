# experiments/nocturnal/trained_models/

Trained model weights and checkpoints for the nocturnal-hypoglycemia
forecasting experiment live here — they are **not** committed. Everything under
this directory is gitignored except this README; training and fine-tuning runs
populate it locally.

Model weights are large run artifacts: GlucoFoundry ships the code to produce
them, not the weights themselves.

## Reproducing

Train or fine-tune a model via the nocturnal recipe / forecasting workflow (see
`experiments/nocturnal/` and `src/workflows/`) to populate this directory.

Paths resolve through the experiment path contract
(`src.experiments.registry.get_experiment_paths`), so setting
`GLUCOFOUNDRY_ARTIFACTS_ROOT` redirects large artifacts to external/scratch
storage without moving the versioned `configs/`.
