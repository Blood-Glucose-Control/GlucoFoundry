# GlucoFoundry — Progress Snapshot

> Human-readable **mirror of the agent's session task tracker**. The live tracker
> is an ephemeral per-session database that is not directly viewable, so this file
> is a **committed snapshot** that is refreshed during work. It is **not
> auto-synced** — treat the "last updated" date as its as-of point.
>
> Companion to [`00_GLUCOFOUNDRY_FOUR_REPO_ROADMAP.md`](./00_GLUCOFOUNDRY_FOUR_REPO_ROADMAP.md)
> (the plan). Phase letters align with task-id prefixes: **A→a, B→b, C→c, D→d**.
>
> Note: `project_tracking.csv` was removed from GlucoFoundry (#42) as old-project
> cruft and is **not** used here; GitHub issues (umbrella #12) are the canonical tracker.

**Last updated:** 2026-11-10 · **Totals:** 42 done · 3 in progress · 13 pending

Status key: ✅ done · 🔄 in progress · ⬜ pending

---

## Phase A — Merge private → public nocturnal (safety net) — ✅ complete

| Task | Status | Ref / notes |
|------|--------|-------------|
| a1 Land unmerged project-2 commits | ✅ | moirai2-moe-lora-sweep decision |
| a2 Pre-merge safety audit (PII/secrets/paths) | ✅ | boundary finding on `private/` + results |
| a2b Public-boundary decision | ✅ | PII cleared (IOBP2 public); promote framework code only |
| a3 Merge-mechanism decision | ✅ | curated path-scoped promotion (not full-history) |
| a4 Merge private → public | ✅ | PR #480 |

## Phase B — Extract GlucoFoundry — 🔄 largely complete

### Core extraction — ✅
| Task | Status | Ref |
|------|--------|-----|
| b1 Create GlucoFoundry repo (full history) | ✅ | — |
| b2 Toolkit charter | ✅ | PR #1 |
| b3a Decouple core + auto-discovery registry | ✅ | PR #20 (closes #3/#4) |
| b3b Per-experiment layout (droppable recipe) | ✅ | PR #15; #13 |
| b3c Scaffold empty artifact dirs | ✅ | PR #22 (closes #5) |
| b3d Relocate results / strip scratch | ✅ | PR #27 |
| b4 Public toolkit docs | ✅ | PR #29 (closes #7) |

### Cleanup / vaporware / rebrand (cross-cutting) — ✅
| Task | Status | Ref |
|------|--------|-----|
| Prune dead standard_forecasting | ✅ | PR #16 |
| Delete dead evaluator.py | ✅ | PR #17 |
| Un-conflate forecasting pipeline from nocturnal | ✅ | PR #25 (closes #18) |
| Remove personalization + dup run-IO | ✅ | PR #26 |
| Strip demo/images/mlflow.db + MLflow vaporware | ✅ | PR #38 (closes #34) |
| Strip docs-internal notebooks | ✅ | PR #41 (closes #35) |
| Remove SLURM tooling + legacy cruft | ✅ | PR #42 (closes #39) |
| Repo identity → GlucoFoundry | ✅ | PR #23 (closes #10) |
| Relicense Apache-2.0 | ✅ | PR #28 (closes #9) |
| Dependency reproducibility policy | ✅ | PR #24 (closes #11) |
| Internal `.venv` + skill rebrand | ✅ | PR #47 (closes #31) |
| Dependency + doc vaporware audit | ✅ | PR #43 (closes #40) |
| Dead-code / own-vaporware audit | ✅ | PR #44, #49 |
| numpy2/pandas3 CI fix | ✅ | PR #479 |
| Tidy issue tracker (close #13, update #12) | ✅ | — |
| Name lock + tracking seed | ✅ | — |

### Remaining in B
| Task | Status | Ref |
|------|--------|-----|
| Repo-path rebrand (old repo name) | ✅ | PR #50 + #52 |
| Register moirai2 in standard wiring | ✅ | PR #62 (closed #46) |
| CGMacros dataset loader | ✅ | PR #58 + #61 |
| Retire old docs hosting (RTD + demo Pages) | ✅ | PR #57 |
| Docs audit | 🔄 | Phase 1 done (PR #60); #56 |
| Experiment-tracking design/impl | 🔄 | #45 (design filed) |
| Fresh-clone reproducibility validation | ⬜ | #8 |
| Hash-locked lockfile | ⬜ | #30 |
| Reproduction recipes (#1/#2) | ⬜ | b4-repro-recipes |

> `scripts/` reorg (#36) and strip-tuned-sweep-configs (#6) are now **sub-tasks**
> of the Experiment-centric restructure below (umbrella #64).

## Experiment-centric restructure (RFC-001) — 🔄 accepted, building

Make each experiment a fully self-contained directory (code + configs + results
+ trained_models + scripts + docs) with a `glucofoundry new-experiment`
scaffolding CLI. See [`RFC-001`](./RFC-001_experiment-centric-structure.md) and
umbrella **#64** (artifacts live inside `experiments/<name>/`; 5 decisions resolved).

| Task | Status | Ref |
|------|--------|-----|
| RFC-001 ratified (5 decisions resolved) | ✅ | PR #63 |
| RFC quick wins (remove dead `configs/models/timegpt/`, progress mirror) | ✅ | PR #65 |
| Phase 2: `ExperimentPaths` resolver + workflow path resolution | ✅ | PR #66 |
| Scrub internal planning refs from `src/` docstrings | ✅ | PR #67 |
| Phase 3: migrate nocturnal self-contained (absorbs #36, part of #6) | ✅ | PR #68 |
| Prune hand-rolled sweep harness; relocate keepers (Optuna pivot) | ✅ | PR #69 |
| Post-prune housekeeping (progress + model-config sweep refs) | ✅ | PR #70 |
| Prune legacy/obsolete scripts (sync, regression-smoke suite) | ✅ | PR #71 |
| Prune forecasting workflow shells; fix stale `scripts/experiments` refs | ✅ | PR #72 |
| Phase 4a: `templates/experiment/` skeleton + interview manifest | ⬜ | #64 |
| Phase 4b: `glucofoundry new-experiment` CLI (full §5 interview + `--from-file`) | ⬜ | #64 |
| Phase 4c: `configs/models/` templates split (baseline→templates, tuned→experiment) | ⬜ | #64 |
| Phase 5: docs/governance update | ⬜ | #64 |

### Small follow-ups (open)
`#53` gluroo DB COPY paths · `#54` skill workspace path · `#55` config-schema index + mkdocs links · `#37` audit scripts.

## Phase C — Carve experiment forks — ⬜ not started
| Task | Status | Ref |
|------|--------|-----|
| c1 Re-express nocturnal as a fork | ⬜ | #32 |
| c2 Carve metabonet into its own fork | ⬜ | #32 |
| c3 Encode core-up / artifacts-down governance | ⬜ | — |

## Phase D — New meal-image project — ⬜ not started
| Task | Status | Ref |
|------|--------|-----|
| d1 Fork GlucoFoundry for meal-image | ⬜ | #33 |
| d2 Multimodal domain-gap analysis | ⬜ | — |
| d3 Experiment + data plan | ⬜ | — |
| d4 Upstream reusable multimodal harness | ⬜ | — |
