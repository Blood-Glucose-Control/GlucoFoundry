# results/

Experiment **run output** lives here — it is **not** committed to the toolkit.

Everything under `results/` is gitignored (except this README); experiment runs
populate it locally. Keeping run artifacts out of version control is deliberate:
GlucoFoundry ships reusable **code**, while the artifacts of any given run belong
to that run (and to the experiment fork that produced them).

## Layout

Experiments write under a per-experiment subdirectory, for example:

```
results/
  nocturnal_forecasting/
    <context>ctx_<horizon>fh/<model>/<timestamp>_<dataset>_<mode>/
      results.json
      ...
```

## Reproducing

Run an experiment recipe (see `experiments/<name>/`) to generate results here,
then summarize them with `scripts/analysis/summarize_experiments.py`.
