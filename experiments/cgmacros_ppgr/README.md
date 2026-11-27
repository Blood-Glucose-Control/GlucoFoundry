# CGMacros post-meal glucose response (PPGR) with frozen CGM representations

> How well do frozen CGM foundation-model embeddings of the 24 h before a meal predict the 2 h post-meal glucose response (GlucoFM paper, Table 4)?

Experiment recipe generated with `glucofoundry new-experiment`. This directory is
**self-contained** — it owns its code, configs, run artifacts, scripts, and docs,
and can be dropped (or carved into its own fork) without touching the toolkit core.

## At a glance

| | |
|---|---|
| **Task family** | `representation_learning` |
| **Target** | `bg_mgdL` |
| **Context / forecast** | `288` / `24` steps (5-min cadence) |
| **Datasets** | cgmacros |
| **Modality** | `unimodal` |
| **Anchoring** | `event_anchored` |
| **Covariates** | — |
| **Models** | cgm_jepa |
| **Metrics** | mae |
| **Holdout** | `patient_level` (held-out fraction `0.2`) |

## Layout

```
cgmacros_ppgr/
  register.py        # registers this recipe with the core
  episodes.py        # build_episodes — meal-anchored events from the raw CSVs
  head.py            # the 2-hidden-layer MLP head trained on frozen features
  eval.py            # evaluate — endpoints + participant-level 5-fold CV
  run.py             # CLI: build events, embed, evaluate, save results.json
  summarize.py       # CgmacrosPpgrSummarizer — runs -> leaderboard CSV
  configs/           # (experiment-owned; none needed yet)
  results/           # run output (gitignored; README only)
  trained_models/    # (unused: heads are retrained per fold)
  scripts/ docs/     # experiment-specific tooling and notes
```

## Task

For every logged meal, predict the glucose change over the next 2 h from the
24 h of CGM before it (GlucoFM paper, App. D.2). Meals are kept only if no
other meal follows within 2 h and both sensors (Dexcom, Libre) cover the 24 h
before and the 2 h after it, so the same meals are scored for each sensor. See
the `episodes.py` docstring for the exact rules.

- **Features:** the frozen CGM-JEPA encoder's mean-pooled embedding of the
  24 h input (96 numbers). Stage `recent_cgm` adds the last hour of glucose.
- **Head:** MLP with two hidden layers (`head.py`), early-stopped on held-out
  training participants. The paper doesn't give head details; ours are in
  `HeadConfig`.
- **Evaluation:** `GroupKFold(5)` by participant, same folds for every seed,
  10 head seeds. Endpoint MAEs: trajectory, positive iAUC, peak rise, peak
  time. Baselines: zero change and the training-fold mean curve.
- **Reported:** mean ± sd over seeds, per sensor and as the two-sensor mean.

## Running

Needs the raw CGMacros CSVs under `cache/data/cgmacros/raw/CGMacros/` and the
`cgm_jepa` extra (`pip install -e '.[cgm_jepa]'`).

```bash
python -m experiments.cgmacros_ppgr.run                      # frozen, cgm_jepa
python -m experiments.cgmacros_ppgr.run --stage recent_cgm
python -m experiments.cgmacros_ppgr.run --checkpoint x_cgm_jepa --participants mets34
```

Meal events are cached in `results/_cache/`; pass `--rebuild-events` to
rebuild them. Each run writes
`results/cgmacros_ppgr_representation_learning/288ctx_24fh/<checkpoint>/<run>/results.json`.
Collect runs into a CSV with:

```python
from experiments.cgmacros_ppgr.summarize import CgmacrosPpgrSummarizer
CgmacrosPpgrSummarizer().summarize(metric="mae")
```

## Differences from the paper

- The paper retrained CGM-JEPA itself ("official checkpoints were
  unavailable"); we use the published Hugging Face weights.
- Head settings are ours (the paper only says "an MLP with two hidden layers").
- Participant set: `all` uses every eligible participant; `mets34` (those with
  a METs column) is a guess at the paper's 34.

## Citation

GlucoFM benchmark (arXiv:2605.30865), CGMacros PPGR task, Appendix D.2 / Table 4.
