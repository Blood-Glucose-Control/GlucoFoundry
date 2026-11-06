# GlucoFoundry Four-Repo Roadmap

**Date:** 2026-10-21 (last updated 2026-11-05)
**Status:** In execution — Phase A complete; Phase B largely complete (see **Progress** below); Phases C/D not started.
**Supersedes scope framing of:** `00_P0-P3_MASTER_REORG_PLAN.md` (that plan targeted
cleaning a single repo; this roadmap reframes the cleanup as *toolkit extraction*).

---

## Vision

Turn this repository lineage into a reusable research toolkit plus a set of
experiment forks. The toolkit is the canonical upstream; each experiment is a
fork that carries only its own experiment layer. Reproducible core flows **up**
to the toolkit; experiment artifacts stay **down** in the forks.

### End state (four repos)

| # | Repo | Role | Scope |
|---|------|------|-------|
| 0 | **GlucoFoundry** | Toolkit / canonical upstream | Reusable, task-general TS-foundation-model core + reproduction recipes. No experiment artifacts. |
| 1 | **Nocturnal-Hypoglycemia** | Fork of GlucoFoundry | Overnight hypoglycemia risk (project #1, NeurIPS lineage). Also retains full historical record. |
| 2 | **Metabonet** | Fork of GlucoFoundry | General 2h-ahead forecasting across all time windows (competition, project #2). |
| 3 | **Meal-Image Forecast** | Fork of GlucoFoundry | Postprandial forecasting from food images; multimodal; horizon up to ~3h (project #3, new). |

### Naming

- Toolkit name: **GlucoFoundry** — a "foundry" forges/casts many artifacts
  (task-general), "found-" nods to **found**ation models, and it is the canonical
  base the forks are cast from.
- Task-general by design: supports forecasting and non-forecasting tasks
  (classification, event/anomaly detection, representation learning, imputation).

---

## Why merge private -> public first (the safety net)

Project #2 (Metabonet) was developed in the private fork
(`private/main`, including `private/metabonet-integration`). As of planning,
`private/main` is **96 commits ahead** of `origin/main` with **0 divergence**
(`origin/main` is the exact merge base). That delta is a mix of genuinely
reusable framework code (sweep/Optuna scaffolding, model adapters, tests) and
experiment artifacts (SUBMISSIONS registries, champion configs, competition
metrics).

Merging private -> public **first** consolidates the complete project record in
one place. Only then is toolkit extraction **risk-free**: GlucoFoundry can strip
aggressively because nothing is ever lost (the full record remains in the
nocturnal repo and its fork).

Note: there are also 4 project-#2 commits on local branch
`feat/moirai2-moe-lora-sweep` not yet in `private/main` (stacker/champion work)
that need a land-or-exclude decision before the merge.

---

## Phases

### Phase A - Merge private -> public nocturnal (safety net)

- `a1-land-feature-commits` - Decide fate of the 4 unmerged `feat/moirai2-moe-lora-sweep` commits (land into `private/main` or exclude).
- `a2-premerge-safety-audit` - PII/PHI, secrets, local-path, raw-data, large-file audit on the 96-commit delta (reuse `pii-secrets-audit` tooling). Gate before public exposure.
- `a3-merge-mechanism-decision` - DECIDED: history-preserving merge commit (no squash). Squash merging is being disabled repo-wide (public + private) because it poisons merge-base tracking across the toolkit/fork topology and destroys the granular record. Rebase optional for small linear PRs.
- `a4-merge-private-to-public` - Open `private/main -> origin/main` PR, run CI, merge, fast-forward public `main`.

### Phase B - Extract GlucoFoundry (risk-free)

- `b1-fork-snapshot-glucofoundry` - Create GlucoFoundry repo from merged nocturnal (carry-history vs clean init).
- `b2-toolkit-charter` - Charter/RFC drawing the core-vs-artifact line.
- `b3-strip-artifacts` - Strip artifacts; finish toolkit-ification items (runtime-core-boundary-lock, repo-root-file-governance-audit, readme-rewrite, experiments-collapse, model-runtime consolidation, reproducibility-lock, license-migration -> Apache-2.0, docs-consolidation).
- `b4-repro-recipes` - Reproduction recipes for experiments #1 and #2 (point at record forks for artifacts).
- `b5-toolkit-validate` - Fresh clone -> install -> run one smoke experiment end-to-end.

### Phase C - Carve experiment forks

- `c1-nocturnal-as-fork` - Re-express nocturnal as GlucoFoundry core + nocturnal experiment layer.
- `c2-metabonet-carveout` - Carve Metabonet into its own GlucoFoundry fork.
- `c3-governance-contract` - Encode core->toolkit / artifacts->fork contract in contributor docs + lightweight CI check.

### Phase D - New Meal-Image project (#3)

- `d1-fork-meal-image` - Fork GlucoFoundry into the meal-image forecast repo.
- `d2-meal-image-domain-gap` - Domain-gap analysis: vision encoder + fusion, image+nutrition+timestamp schema, ~3h postprandial horizon, new metrics; separate reused-harness vs net-new.
- `d3-meal-image-exp-data-plan` - Draft experiment + data plan (datasets, labeling, augmentation, evaluation).
- `d4-upstream-multimodal` - Contribute reusable multimodal harness pieces back up to GlucoFoundry.

---

## Critical path and open decisions

- Critical path: **A -> B -> (C parallel with D)**. Phases A and B are
  essentially done; the next ready work is Phase C (`c1-nocturnal-as-fork`).
- Open decisions to confirm during execution:
  1. Land vs exclude the 4 `feat/moirai2-moe-lora-sweep` commits.
  2. (DECIDED) Private -> public merge uses a history-preserving merge commit,
     not squash. Squash merging is being disabled repo-wide.
  3. GlucoFoundry: carry full history vs clean `init`.

---

## Progress (as of 2026-11-05)

- **Phase A — complete.** Private -> public nocturnal merge landed (stranded
  commits via private PR; curated path-scoped promotion to public). GlucoFoundry
  repo created carrying full history.
- **Phase B — largely complete.** Charter (#1/#2), core decouple + auto-discovery
  (#3), `episode_builders` split (#4), droppable top-level recipes (#13), scaffold
  artifact dirs (#5), public toolkit docs (#7), Apache-2.0 relicense (#9), repo
  identity (#10), dependency policy (#11). Additional cleanup beyond the original
  scope: stripped MLflow/demo/images/docs-internal/SLURM/legacy cruft (#38/#41/#42),
  dependency + doc **vaporware audit** (#40), **dead-code cleanup** (#44/#49),
  `.venv` + skill rebrand (#31).
  - Remaining in B: `b4-repro-recipes` (partial — the nocturnal recipe exists;
    moving experiment-tuned sweep configs tracked in #6/#36) and
    `b5-toolkit-validate` fresh-clone validation (#8). Open follow-ups: experiment
    tracking (#45), moirai2 registration (#46), repo-path rebrand (#48),
    lockfile (#30).
- **Phase C — not started** (fork carve-out, #32).
- **Phase D — not started** (meal-image project, #33).

## Tracking

- Phase labels and task-id prefixes are aligned: **A->a, B->b, C->c, D->d**.
- `project_tracking.csv` was **removed** from GlucoFoundry (#42) as old-project
  cruft — it is no longer the tracking source. Live tracking now lives in the
  GitHub issues (umbrella **#12** for Phase B) plus the ephemeral session task DB.
