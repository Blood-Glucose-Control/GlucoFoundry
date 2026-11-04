# trained_models/

Trained model weights and checkpoints live here — they are **not** committed to
the toolkit.

Everything under `trained_models/` is gitignored (except this README). Training
and fine-tuning runs populate it locally. Model weights are large run artifacts;
GlucoFoundry ships the code to produce them, not the weights themselves.

## Reproducing

Train or fine-tune a model via its recipe / workflow (see `experiments/<name>/`
and `src/workflows/`) to populate this directory.
