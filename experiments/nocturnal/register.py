# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Registration hook for the nocturnal experiment recipe.

Auto-discovered by the experiment registry (``src/experiments/registry.py``). The
core resolves this experiment by name and never imports the recipe directly, so
the ``experiments/nocturnal/`` directory can be dropped cleanly.
"""

from experiments.nocturnal.eval import evaluate_nocturnal_forecasting
from experiments.nocturnal.summarize import NocturnalSummarizer
from src.evaluation.episode_builders import build_midnight_episodes
from src.experiments.registry import ExperimentSpec, register_experiment

register_experiment(
    ExperimentSpec(
        name="nocturnal",
        build_episodes=build_midnight_episodes,
        summarizer=NocturnalSummarizer,
        evaluate=evaluate_nocturnal_forecasting,
        description="Overnight (midnight-anchored) hypoglycemia forecasting.",
    )
)
