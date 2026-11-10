# templates/

Scaffold templates used by the `glucofoundry new-experiment` CLI.

- [`experiment.yaml`](./experiment.yaml) — the **interview manifest**: the set of
  questions the scaffolder asks (or reads from a `--from-file` answers file) and
  the tokens those answers map to. This file is metadata; it is **not** copied
  into a generated experiment.
- [`experiment/`](./experiment) — the **scaffold content** that is rendered into
  `experiments/<name>/`. Files ending in `.tmpl` are rendered (tokens replaced,
  the `.tmpl` suffix stripped); every other file is copied verbatim.

## Token syntax

Templated files use `{{ token }}` placeholders whose names match the `key`
fields in `experiment.yaml`. A few tokens are **derived** by the CLI rather than
asked directly:

- `summarizer_class` — PascalCase class name derived from `experiment_name`.
- `experiment_type` — `"<experiment_name>_<task_family>"`, the run-artifact
  subdirectory an experiment's summarizer scans.

See `experiment.yaml` for the authoritative token list, types, and defaults.
