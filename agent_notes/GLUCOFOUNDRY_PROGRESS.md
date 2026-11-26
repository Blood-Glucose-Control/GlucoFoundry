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

**Last updated:** 2026-11-25 · **Totals:** 55 done · 2 in progress · 11 pending

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
| RFC-001 status → Accepted (all 5 decisions since shipped via #66/#68/#74/#75) | ✅ | PR #86 |
| RFC quick wins (remove dead `configs/models/timegpt/`, progress mirror) | ✅ | PR #65 |
| Phase 2: `ExperimentPaths` resolver + workflow path resolution | ✅ | PR #66 |
| Scrub internal planning refs from `src/` docstrings | ✅ | PR #67 |
| Phase 3: migrate nocturnal self-contained (absorbs #36, part of #6) | ✅ | PR #68 |
| Prune hand-rolled sweep harness; relocate keepers (Optuna pivot) | ✅ | PR #69 |
| Post-prune housekeeping (progress + model-config sweep refs) | ✅ | PR #70 |
| Prune legacy/obsolete scripts (sync, regression-smoke suite) | ✅ | PR #71 |
| Prune forecasting workflow shells; fix stale `scripts/experiments` refs | ✅ | PR #72 |
| Phase 4a: `templates/experiment/` skeleton + interview manifest | ✅ | PR #73 |
| Phase 4b: `glucofoundry new-experiment` CLI (full §5 interview + `--from-file`) | ✅ | PR #74 |
| Phase 4c: `configs/models/` templates split (baseline→templates, tuned→experiment) | ✅ | PR #75 |
| Refresh workflow diagrams (de-MLflow main diagram + render 3 new workflow SVGs) | ✅ | PR #85 |
| Phase 5: docs/governance update | ⬜ | #64 |

### Phase 4 review finding → Optuna-first redesign (umbrella #76)

Reviewing Phase 4 surfaced that the scaffold template is **forecasting-rigid +
single-model-centric**, and the Optuna infra has anti-patterns + a core/experiment
boundary violation (the "nocturnal" Optuna config actually runs on MetaboNet data).
Workflow diagrams are in [`agent_notes/diagrams/`](./diagrams). Tracked under
umbrella **#76**. **R1–R6 are complete** (the engine, anti-pattern fixes, the
core-vs-experiment reorg, nocturnal cleanup, nocturnal's own sweep, and the
Optuna-first scaffolder); **R7 (docs) remains**.

| Task | Status | Ref |
|------|--------|-----|
| R1 extract a neutral shared Optuna engine (out of `chronos2/`) | ✅ | PR #90 (#77) |
| R2 unify pruning/reporting + parallel-safe storage | ✅ | #78 — (c) metrics guard PR #93 + (a) parallel-safe storage PR #94 + (b) chronos2 `fit()` monkey-patch→`ReportingChronos2Model` subclass PR #101 (review addressed) |
| R3 reorganize HPO: engine-only core vs MetaboNet-2026 experiment sweep tree | ✅ | PR #91 (rename) + #92 (honest move); #79 closed |
| R4 clean house on `experiments/nocturnal/` (archive rebuttal/grand-summary/viz cruft) | ✅ | PR #87 (#80) |
| R5 stand up nocturnal's Optuna on the reported core metrics | ✅ | PR #96 — `experiments/nocturnal/hpo/`, 10 families on the shared engine; naive+statistical real-run validated |
| R6 Optuna-first scaffold + de-forecasting the stubs (revises #64) | ✅ | PR #103 (#82) — sweep is the default generated workflow; task-general episodes/eval stubs |
| R7 Optuna/scaffolder/single-model workflow docs | ⬜ | #83 |

> **R3 deviation (shipped):** the original plan decoupled per-model
> `objective`/`search_space` into experiment-agnostic skeletons kept in core. The
> code proved fully MetaboNet-specific (not thin skeletons), so with maintainer
> sign-off R3 took the **honest move**: only `engine.py` stays in core and the
> entire family tree moved to `experiments/metabonet_2026/hpo/`. This obviated the
> originally-planned R3c (decouple + move specs). See #79.

> **R5 follow-ups:** #97 tamborlane_2008 processing bug (✅ done, PR #106) · #98
> remove manual ablation YAML grids (now unblocked — sweeps proven; still open) ·
> #99 validate deep/foundation family real runs on GPU (✅ done — all 8 families
> smoke-validated on Blackwell; fixes in PR #107) · #100 MetaboNet-2026 HPO
> storage under the experiment dir (✅ done, PR #105).
>
> **Issue housekeeping (2026-11-25):** closed the R2 (#78) and R5 (#81) umbrella
> sub-issues now that all their PRs merged (R2: #93/#94/#101; R5: #96). R6 (#82)
> and the #97/#100/#99 follow-ups auto-/closed on their PR merges
> (#103/#106/#105/#107); #108 (moirai Blackwell torch) closed via #107.

### Other open items (for pickup once this phase settles)

- **Optuna-first redesign:** umbrella **#76** → R1–R7 (**#77–#83**) — see the table above.
- **Docs / usability:** `#55` config-schema index + mkdocs links · `#56` docs audit + Pages · `#54` skill workspace path · `#53` gluroo DB COPY paths.
- **Reproducibility:** `#8` fresh-clone smoke validation · `#30` hash-locked lockfile · `#45` experiment tracking (MLflow replacement).
- **Forks (Phase C/D):** `#32` nocturnal + metabonet forks · `#33` meal-image fork.
- **Likely already done — candidates to close:** `#6` strip/move tuned sweep configs (done via #68/#69/#75) · `#36` scripts reorg by core-vs-experiment (done via #68/#69/#72) · `#37` audit scripts for nocturnal assumptions (done via #71/#72).

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
