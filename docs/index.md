# GlucoFoundry

**Glucose Time-Series Foundation Models — a task-general, reusable research toolkit.**

GlucoFoundry is the canonical upstream for a family of glucose time-series
research projects. It provides the reusable machinery — data loaders and schemas,
a model-adapter layer over many time-series foundation models, an Optuna
sweep framework, evaluation metrics, and workflow orchestration — while each
experiment lives as a self-contained, **droppable recipe**.

Fork GlucoFoundry to start a new experiment; contribute genuinely reusable
improvements back upstream. The toolkit ships **code**, not experiment artifacts.

## Layout

```
src/            core (experiment-agnostic): data, models, workflows, evaluation
experiments/    droppable, self-contained experiments (e.g. experiments/nocturnal/):
                each owns its configs/, scripts/, results/, trained_models/
configs/        toolkit-generic config templates (data, models, evaluation)
scripts/        canonical experiment-agnostic CLI/sweep entry points
tests/          core + recipe tests
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pytest --ignore=tests/models
```

Model-family tests use dedicated environments — see the
[Contributing guide](https://github.com/Blood-Glucose-Control/GlucoFoundry/blob/main/CONTRIBUTING.md)
and the `Makefile` (`make test-<family>`).

## Architecture

- [Toolkit Architecture](architecture/toolkit-architecture.md) — core, droppable
  recipes, and the experiment registry
- [Repository Design Handbook](architecture/repository-design-handbook.md)
- [Workflow Config Schemas](architecture/workflow-configs-schema-guide.md)
- [Config Schema Artifacts](architecture/config-schema-artifacts.md)
- [Model Implementation Quality Checklist](architecture/model-implementation-quality-checklist.md)

## Documentation

This site is built with [mkdocs.org](https://www.mkdocs.org) using the Material
theme (`mkdocs serve` to preview locally, `mkdocs build` to build).
