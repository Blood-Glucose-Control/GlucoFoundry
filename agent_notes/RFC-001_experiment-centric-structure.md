# RFC-001 — Experiment-centric repository structure + scaffolding CLI

**Status:** Accepted — §7 decisions resolved 2026-11-08
**Date:** 2026-11-08
**Tracking:** umbrella issue (reframes #36, #6)
**Decisions locked by maintainer:** lean into experiment-centric organization;
run artifacts live **inside** each experiment (`experiments/<name>/results/` and `experiments/<name>/trained_models/`).

---

## 1. Problem

Today `experiments/<name>/` holds only recipe *code* (register/episodes/eval/
summarize). The spaces that an experiment actually *uses* — `configs/`,
`results/`, `trained_models/`, `scripts/` — are **shared at the repo root** with
only partial namespacing (`configs/experiments/nocturnal_forecast/`,
`results/nocturnal_forecasting/`, `results/optuna/...`).

Consequences:
- Multiple experiments collide in shared `results/` / `trained_models/`, making
  it hard to tell which run belongs to which experiment.
- Experiment-tuned configs sit next to toolkit-generic ones with no clear line.
- "Dropping" an experiment leaves orphaned configs/results behind — it is not
  actually self-contained.

## 2. Decision

Make each experiment a **fully self-contained directory** that owns its code,
configs, artifacts, scripts, and docs. The toolkit **core** (`src/`) stays pure
and reusable and flows **up**; experiment directories stay **down** in forks.

This strengthens the four-repo governance (core-up / artifacts-down) and makes
experiments genuinely droppable: `rm -rf experiments/<name>/` removes everything.

## 3. The generic-vs-specific boundary (the crux)

| Stays **toolkit core** (flows up) | Moves into **`experiments/<name>/`** (stays down) |
|---|---|
| `src/data/` loaders, `src/models/` families, `src/config/schemas/` **schemas**, `src/workflows/` framework, sweep engine | recipe code: `register.py`, `episodes.py`, `eval.py`, `summarize.py` |
| generic entrypoints in `scripts/` (workflow runner, sweep launcher) | experiment-specific scripts (summaries, figures, rebuttal analyses) |
| `templates/experiment/` (scaffold template + baseline configs) | **tuned** model/sweep/holdout **config values** |
| dataset registry (which datasets exist + their schemas) | `results/`, `trained_models/` (this experiment's runs); `docs/`, `notebooks/` |

Key nuance for `configs/models/<family>/`: the **schemas** are core; the **tuned
values** are experiment artifacts. The toolkit ships *baseline templates*; each
experiment owns its *tuned copies*.

## 4. Target structure

```
GlucoFoundry/
  src/                       # pure toolkit core — flows UP
  scripts/                   # ONLY toolkit-generic entrypoints
  templates/experiment/      # cookiecutter template (baseline configs + stubs)
  experiments/
    <name>/
      register.py  episodes.py  eval.py  summarize.py
      configs/ { data/  models/  sweeps/ }   # experiment-tuned
      results/            # gitignored (README only)
      trained_models/     # gitignored (README only)
      scripts/            # experiment-specific
      docs/  notebooks/  README.md
```

Root `results/` and `trained_models/` are removed (or demoted to a thin default
for ad-hoc runs only). `src/workflows` resolves an **experiment root** and writes
artifacts under `experiments/<name>/` (the pipeline already threads
`--experiment`, so this is path resolution, not a rewrite).

## 5. Scaffolding CLI — `glucofoundry new-experiment`

A cookiecutter-style generator that stamps `experiments/<name>/` from
`templates/experiment/` and pre-wires `register.py` from the answers.

Questions asked at experiment start:

- **Identity:** name (slug) + title + one-line research question.
- **Task family:** forecasting / event-detection / classification / imputation /
  representation-learning.
- **Target & horizon:** target var (default `bg_mM`), horizon (e.g. 2h/3h) or
  event type; sampling interval (default 5-min).
- **Datasets:** multi-select from registered loaders (aleppo_2017, brown_2019,
  lynch_2022, tamborlane_2008, gluroo, metabonet, cgmacros).
- **Modality:** unimodal CGM vs multimodal (+ food images / nutrition).
- **Episode / anchoring strategy:** whole-series / sliding-window /
  time-anchored (midnight → nocturnal) / event-anchored (postprandial →
  meal-image); context + prediction length.
- **Covariates:** IOB, COB, insulin-availability, carbs, …
- **Models:** subset of `SUPPORTED_MODEL_TYPES`; zero-shot vs fine-tune default.
- **Evaluation:** metrics (RMSE/MAE, DTW/shape, PIT/CRPS calibration, event
  metrics like hypo recall) + holdout strategy (patient-level %, temporal,
  nested CV) + split ratios.
- **Extras:** include Optuna sweep configs? example notebooks? license/citation.

Output: the full self-contained directory with `register.py` wired (episode
builder + metrics + datasets + models from the answers).

## 6. Phased plan

1. **RFC / decision** (this doc) — ratify structure + boundary; update charter/roadmap.
2. **Experiment contract** — define the dirs/files an experiment owns; make
   `src/workflows` resolve experiment-relative `results/trained_models/configs`.
3. **Migrate nocturnal** — move its configs/results/scripts into
   `experiments/nocturnal/` (absorbs #36 and part of #6).
4. **Scaffolder** — build `glucofoundry new-experiment` + `templates/experiment/`.
5. **Docs + governance** — update `docs/architecture/*` with the ratified
   structure and the experiment contract.

Housekeeping found alongside this: delete the stale `nocturnal_forecast.egg-info/`
(leftover from the old package name).

## 7. Resolved decisions

1. **Baseline model configs — templates only.** Ship **one clean baseline per
   model family** under `templates/experiment/configs/models/` (the template is
   the browsable reference). The current ~20 tuned variants per family in
   `configs/models/<family>/` are nocturnal artifacts and move to
   `experiments/nocturnal/configs/models/`. No separate root gallery.
2. **Root `results/` / `trained_models/` — removed.** All run artifacts live under
   `experiments/<name>/results/` and `experiments/<name>/trained_models/`. Scratch
   work uses a scratch experiment rather than a shared root default.
3. **Data holdout configs — per-experiment, seeded from templates.** Reusable
   holdout *strategy* templates live in `templates/experiment/configs/data/`; the
   concrete per-experiment split (datasets + seed) lives in
   `experiments/<name>/configs/data/`.
4. **Scaffolding CLI — `glucofoundry new-experiment`.** A console entry point
   (not a `make` target), **interactive by default** with a non-interactive
   **`--from-file spec.yaml`** mode (the spec becomes a committed, reviewable
   artifact). Implemented with a **lightweight generator** (Jinja2 / string
   templates over `templates/experiment/`) — no cookiecutter dependency.
5. **Workflow path contract — `ExperimentPaths` resolver.** Add a resolver keyed
   off the experiment name in `src/experiments/registry.py` (alongside
   `ExperimentSpec`): `experiments/<name>/{results,trained_models,configs}/`,
   resolved **repo-root-relative** with an env/flag override for large/scratch
   storage. The pipeline's `--experiment` + `output_dir` default to the
   experiment's `results/`.

## 8. Relation to existing issues

- **#36** (scripts reorg) and **#6** (move tuned sweep configs) become **sub-tasks**
  of this restructure rather than standalone efforts.
- **#32 / #33** (nocturnal/metabonet/meal-image forks) benefit directly: each fork
  becomes `toolkit core + one self-contained experiment`.
