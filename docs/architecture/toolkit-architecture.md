# GlucoFoundry architecture

GlucoFoundry is a **task-general, reusable toolkit** for glucose time-series
foundation-model research. Its guiding principle:

> **Toolkit core + droppable experiment recipes + empty artifact scaffolding.**
> Reproducible code flows *up* into the toolkit; experiment artifacts stay *down*
> in the forks.

Fork GlucoFoundry to run an experiment; contribute genuinely reusable
improvements back upstream. The installed library ships **code**, not artifacts.

## Layout

```
src/            core (experiment-agnostic): data, models, workflows, evaluation
experiments/    droppable experiment recipes (e.g. experiments/nocturnal/)
configs/        config templates (data, models, evaluation)
scripts/        canonical CLI/sweep entry points
tests/          core + recipe tests
results/        run output — gitignored (populated locally)
trained_models/ model weights — gitignored (populated locally)
```

## The core is experiment-agnostic

The hard rule that makes the toolkit modular:

> **No module under `src/` imports from `experiments/`.** The dependency
> direction is one-way: experiment → core.

The core (data loaders/schemas, the model-adapter layer, the Optuna sweep
framework, evaluation metrics, and the forecasting/runtime workflows) knows
nothing about any specific experiment. It resolves experiment-specific behavior
**by name** through the experiment registry.

You can verify the boundary at any time:

```bash
git grep "from experiments" -- src   # should return nothing
```

## Experiments are droppable recipes

Each experiment is a single self-contained directory under `experiments/<name>/`
that imports the pure `src/` core and **self-registers**. Deleting the directory
removes the experiment cleanly, with no edits to core code.

A recipe typically contains:

```
experiments/<name>/
    register.py   # registration hook (auto-discovered)
    eval.py       # experiment-specific evaluation / metrics / anchoring
    summarize.py  # experiment result summarizer
    run_evaluation.py  # CLI entry point
    episodes.py   # experiment-specific episode construction (if any)
    README.md     # how to reproduce this experiment
```

The shipped example is `experiments/nocturnal/` — overnight (midnight-anchored)
hypoglycemia forecasting. These recipes are **example pipelines**: they are not
run inside the toolkit; a user runs them to generate their own artifacts in
`results/`.

## The experiment registry

The core and recipes meet at the registry in
[`src/experiments/registry.py`](../../src/experiments/registry.py).

### Declaring an experiment

A recipe's `register.py` registers an `ExperimentSpec`:

```python
from src.experiments.registry import ExperimentSpec, register_experiment
from experiments.nocturnal.episodes import build_midnight_episodes
from experiments.nocturnal.eval import evaluate_nocturnal_forecasting
from experiments.nocturnal.summarize import NocturnalSummarizer

register_experiment(
    ExperimentSpec(
        name="nocturnal",
        build_episodes=build_midnight_episodes,
        summarizer=NocturnalSummarizer,
        evaluate=evaluate_nocturnal_forecasting,
        description="Overnight (midnight-anchored) hypoglycemia forecasting.",
    )
)
```

All hooks are optional, so a recipe opts into only what it provides.

### Auto-discovery

At startup the registry scans `experiments/*/register.py` and imports each to
trigger registration. Because discovery is path-based, dropping a recipe
directory simply removes that experiment — no edits elsewhere. A broken recipe is
logged and skipped, not fatal.

### Resolving by name

Core code resolves experiment behavior by name, never by importing a recipe:

```python
from src.experiments.registry import get_episode_builder

build_episodes = get_episode_builder("nocturnal")  # non-optional callable
```

The forecasting workflow takes the experiment name from its run config
(`--experiment`, default `nocturnal`), so the generic pipeline is not implicitly
tied to any one experiment.

## Artifact scaffolding

`results/` and `trained_models/` ship as **empty but present** directories (a
README only); everything else under them is gitignored. The toolkit ships the
code to produce artifacts, not the artifacts. Experiment output is written under
`results/<experiment>/…` locally.

## Adding a new experiment

1. Create `experiments/<name>/` and import only from `src/`.
2. Provide the experiment-specific pieces (episode builder, evaluation,
   summarizer, CLI entry point) as needed.
3. Add `experiments/<name>/register.py` registering an `ExperimentSpec`.
4. Run it to generate artifacts under `results/<name>/`.

Keep the dependency direction one-way (experiment → core); if you find yourself
wanting the core to import your experiment, that logic probably belongs in the
core as a generic capability instead.
