## Model Benchmarking <a id="model-benchmarking"></a>

GlucoFoundry benchmarks models with **per-model-family Optuna sweeps** built on a
shared, experiment-agnostic engine (`src/workflows/hpo/engine.py`). Each
experiment owns its sweep tree under `experiments/<name>/hpo/<family>/`, and every
trial runs the single-model training/evaluation loop with hyperparameters sampled
from that family's search space.

See the [Research workflows](../../architecture/workflows.md) page for the sweep
diagram, and the [example scripts guide](../example-scripts-guide.md) for the
launchers and the canonical forecasting workflow.

## Running a sweep

Sweeps are launched per family; workers cooperate on one shared Optuna study. For
the shipped nocturnal recipe:

```bash
experiments/nocturnal/scripts/sweeps/optuna/launch_nocturnal_sweep.sh <family>
```

Each family defines its `MODEL_TYPE`, a `DEFAULT_CONFIG`, and a search space under
`experiments/<name>/hpo/<family>/`. The engine stores trials in a shared Optuna
`JournalStorage` under the experiment's own `results/`, and the experiment's
summarizer turns completed trials into the best configuration and a leaderboard.
