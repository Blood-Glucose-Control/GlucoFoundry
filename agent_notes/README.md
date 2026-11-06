# Agent Notes Directory

Internal planning / status notes for the **GlucoFoundry** toolkit-extraction
project. Public docs live in `docs/`; this folder is internal and may hold
phase/task tracking detail.

## Current (GlucoFoundry project)

- `00_GLUCOFOUNDRY_FOUR_REPO_ROADMAP.md` — active cross-repo roadmap (extract the
  GlucoFoundry toolkit + 3 experiment forks). Phases A/B/C/D align with task-id
  prefixes a/b/c/d.
- `GLUCOFOUNDRY_PROGRESS.md` — human-readable snapshot of the agent's session task
  tracker (status + PR/issue refs); refreshed during work, not auto-synced.
- `diagrams/` — general-audience GlucoFoundry workflow flowchart (Mermaid source,
  matplotlib render scripts, exported PNG/SVG) + explainer plots.

## Archived (previous nocturnal / metabonet reorg — kept for the record)

Preserved, not deleted. These predate the GlucoFoundry extraction and used the
earlier single-repo `P#-##` phase system.

- `archive_nocturnal-reorg/` — the P0–P3 reorg notes: master reorg plan,
  PII/data-inventory audits, backup/branch inventories, private-fork governance,
  MLflow/Optuna + pydantic-schema + model-runtime execution plans, workspace/
  pylance handoffs, Ruff baseline cleanup, retention reports, and the
  `script_cleanup/` inventories.
- `archive_2026-08-07/`, `archive_2026-08-18/` — earlier superseded handoff/plan
  variants.

These are historical traceability only — not current runbooks.

## Conventions

- Current GlucoFoundry tracking lives in GitHub issues (umbrella #12) plus the
  session task DB (mirrored to `GLUCOFOUNDRY_PROGRESS.md`). `project_tracking.csv`
  was removed (#42) and is no longer used.
- Archived notes use the legacy `P#-##` prefix (P0/P1 = phases of the prior
  nocturnal/metabonet reorg).
- Preserve history: archive obsolete notes instead of deleting them.
