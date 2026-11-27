# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Registration hook for the cgmacros_ppgr experiment recipe.

Auto-discovered by the experiment registry (``src/experiments/registry.py``). The
core resolves this experiment by name and never imports the recipe directly, so
the ``experiments/cgmacros_ppgr/`` directory can be dropped cleanly.
"""

from experiments.cgmacros_ppgr.summarize import CgmacrosPpgrSummarizer
from src.experiments.registry import ExperimentSpec, register_experiment

# Only the summarizer is registered. The forecasting workflow calls
# ``build_episodes`` on one patient's 5-min DataFrame and scores a forecaster's
# predictions; this task needs all participants' raw readings and trains a new
# head per fold, so it runs through ``run.py`` instead.
register_experiment(
    ExperimentSpec(
        name="cgmacros_ppgr",
        summarizer=CgmacrosPpgrSummarizer,
        description="CGMacros post-meal glucose response (PPGR) with frozen CGM representations",
    )
)
