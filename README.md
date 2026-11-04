# GlucoFoundry

**Glucose Time-Series Foundation Models — a task-general, reusable research toolkit.**

GlucoFoundry is the canonical upstream for a family of glucose time-series
research projects. It provides the reusable machinery — data loaders and schemas,
a model-adapter layer over many time-series foundation models, an Optuna/MLflow
sweep framework, evaluation metrics, and workflow orchestration — while each
experiment lives as a self-contained, **droppable recipe**.

Fork GlucoFoundry to start a new experiment; contribute genuinely reusable
improvements back upstream. The toolkit ships **code**, not experiment artifacts.

## How it works

```
src/            core (experiment-agnostic): data, models, workflows, evaluation
experiments/    droppable experiment recipes (e.g. experiments/nocturnal/)
results/        run output — gitignored (populated locally)
trained_models/ model weights — gitignored (populated locally)
```

- The **core never imports experiments**. Experiments self-register via
  auto-discovery (`experiments/<name>/register.py`) and are resolved by name, so
  cloning the toolkit and deleting an experiment directory removes it cleanly.
- Each experiment recipe bundles its evaluation, visualization, summarization,
  and run entry point, importing only the pure `src/` core.

See the design reference in
[`docs/design/toolkit-charter.md`](docs/design/toolkit-charter.md).

## Experiment projects

GlucoFoundry is forked to run specific experiments, for example:

- **Nocturnal hypoglycemia** — overnight (midnight-anchored) hypoglycemia
  forecasting (`experiments/nocturnal/`).
- **Metabonet** — general 2-hour-ahead forecasting across all time windows.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest --ignore=tests/models
```

Model-family tests use dedicated environments — see the `Makefile`
(`make test-<family>`) and `CONTRIBUTING.md`.

## License

This project is currently licensed under a custom research license based on
CC BY-NC 4.0. The software is freely available for academic institutions and
non-profit research organizations; commercial use requires a separate commercial
license.

See [LICENSE](LICENSE) for full details.

**For commercial licensing inquiries, please contact:** christopher@gluroo.com

### License Summary
- Free for research — Academic institutions and non-profit research organizations
- Attribution required — Please cite Blood-Glucose-Control in your work
- Commercial use restricted — Requires separate commercial license
- Commercial licensing — Contact us for business use
