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

GlucoFoundry is licensed under the **Apache License 2.0**. You may use, modify,
and distribute it — including for commercial purposes — provided you retain the
copyright notice and license, state significant changes, and comply with the
license terms.

See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full details.

- Free to use, modify, and distribute (including commercially)
- Attribution required — retain the copyright/`NOTICE` and license text
- Includes an explicit patent grant
- Provided "as is", without warranty
