# Contributing to GlucoFoundry

Thank you for your interest in contributing to **GlucoFoundry** — a task-general,
reusable toolkit for glucose time-series foundation-model research.

GlucoFoundry is **fork-first**: it provides the reusable core (data, models,
sweeps, evaluation, workflows), and each experiment lives as a self-contained,
droppable recipe under `experiments/<name>/`. Contributions that make the core
more reusable are especially welcome.

## Quick Start

1. **Fork and clone** the repository
2. **Set up environment**: `python -m venv .venv && source .venv/bin/activate`
3. **Install dependencies**: `pip install -e .`
4. **Run tests**: `pytest --ignore=tests/models`
5. **Create a feature branch**: `git checkout -b feat/your-feature`
6. **Make changes** and commit with conventional-commit messages
7. **Submit a pull request**

## Environment Setup

GlucoFoundry uses **two types of virtual environments**:

### Core development (`.venv`)
For the core toolkit, experiments, tests, and data work:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Model-specific environments (`.venvs/<model>/`)
Different foundation models require mutually conflicting package versions
(e.g., `transformers`), so each model family has an isolated environment used for
training and inference:

```bash
# Set up and activate a model environment (creates the venv on first run)
source scripts/setup_model_env.sh ttm
source scripts/setup_model_env.sh sundial

# Deactivate when done
deactivate
```

Model families are defined in `pyproject.toml` under
`[project.optional-dependencies]`. To add one, add an entry there and run
`source scripts/setup_model_env.sh <model>`. Model-family tests run via the
`Makefile` (`make test-<family>`).

**When to use which:**
| Task | Environment |
|------|-------------|
| Core toolkit / experiment development | `.venv` |
| Running the non-model test suite | `.venv` |
| Model training/inference (e.g. TTM) | `source scripts/setup_model_env.sh <model>` |

## Dependency policy

GlucoFoundry is a fork-first research toolkit, so dependencies favour
**compatibility over hard pins**:

- **Abstract dependencies** in `pyproject.toml` use **compatible ranges** — a
  floor for features we rely on, and a **ceiling just below the next untested
  major** for the core scientific stack (for example `numpy>=2.0,<3`,
  `pandas>=2.3,<4`). The ceilings exist specifically to keep a surprise major
  release from silently breaking CI; raise them deliberately once the new major
  is validated.
- **Python**: 3.12 is the canonical/CI version (`requires-python = ">=3.12"`).
- **Model-family environments are isolated** (`.venvs/<model>/`) because
  foundation models require mutually conflicting dependency sets; they are not
  expected to share a single resolved environment with the core.
- **Exact, hash-locked reproduction** (a lockfile) is intentionally **not** in
  place yet: a single lockfile cannot represent the multiple isolated model-family
  venvs or the CPU/GPU `torch` split, so it is tracked as dedicated future work.

When bumping a core dependency across a major version, validate
`pytest --ignore=tests/models` on the target version before raising the ceiling.

## What can you contribute?

### Core toolkit
- New foundation-model adapters (via the `ModelAdapter` protocol)
- Evaluation metrics and probabilistic/shape scoring
- Sweep-framework and workflow improvements
- Data loaders and schema adapters for new datasets

### Experiments (recipes)
- New experiment recipes under `experiments/<name>/` (see the architecture guide)
- Reproducibility improvements to existing recipes

### Infrastructure & docs
- CI/CD, caching, logging, and developer-workflow improvements
- Tutorials, guides, and architecture documentation

## Adding a new experiment

Experiments are **droppable recipes** that import the pure `src/` core and
self-register, so the core never imports them. See
[docs/architecture/toolkit-architecture.md](docs/architecture/toolkit-architecture.md)
for the full design. In short, a recipe lives at `experiments/<name>/` and
provides a `register.py` that registers an `ExperimentSpec` (its episode builder,
summarizer, and evaluation entry point). Deleting the directory removes the
experiment cleanly.

## Development guidelines

- **Use conventional commits**: `feat:`, `fix:`, `docs:`, `chore:`, etc.
- **Add tests** for new functionality.
- **Keep the core experiment-agnostic**: `src/` must never import from
  `experiments/`.
- **Run formatting/linting**: `ruff format .` and `ruff check .` (repo selection
  is `E4,E7,E9,F,I`). Run `pre-commit run --all-files` before pushing — CI lints
  the whole repo.
- **Configure Pylance via `pyrightconfig.json`**, not VS Code
  `diagnosticSeverityOverrides`.

## Code quality standards

- ✅ Tests pass (`pytest --ignore=tests/models`; model families via `make test-<family>`)
- ✅ Lint clean (`ruff check`, `ruff format`)
- ✅ Type hints on public function signatures
- ✅ Docstrings for public functions
- ✅ No merge conflicts with `main`

## Issue reporting

- **Bugs**: open an issue describing the problem and how to reproduce it
- **Features**: open an issue describing the proposed capability
- **Questions**: use GitHub Discussions

## License

By contributing, you agree that your contributions are licensed under the
project's [Apache License 2.0](LICENSE).

---

For a deeper developer guide, see [docs/contributing.md](docs/contributing.md).
