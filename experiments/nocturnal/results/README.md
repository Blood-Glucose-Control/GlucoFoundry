# experiments/nocturnal/results/

Run **output** for the nocturnal-hypoglycemia forecasting experiment lives here
— it is **not** committed. Everything under this directory is gitignored except
this README; experiment runs populate it locally.

Keeping run artifacts out of version control is deliberate: GlucoFoundry ships
reusable **code**, while the artifacts of any given run belong to that run (and
to the experiment fork that produced them).

## Layout

Runs write under a per-configuration subtree, for example:

```
results/
  nocturnal_forecasting/
    <context>ctx_<horizon>fh/<model>/<timestamp>_<dataset>_<mode>/
      results_summary.json
      ...
```

## Reproducing

Run the nocturnal recipe (`experiments/nocturnal/run_evaluation.py` or the
forecasting workflow with `--experiment nocturnal`) to generate results here,
then summarize them with
`experiments/nocturnal/scripts/analysis/summarize_experiments.py`.

Paths resolve through the experiment path contract
(`src.experiments.registry.get_experiment_paths`), so setting
`GLUCOFOUNDRY_ARTIFACTS_ROOT` redirects large artifacts to external/scratch
storage without moving the versioned `configs/`.
