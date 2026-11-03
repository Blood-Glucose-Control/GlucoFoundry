# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Nocturnal (midnight-anchored) hypoglycemia forecasting experiment recipe.

Self-contained recipe: evaluation (``eval``), summarization (``summarize``),
episode/holdout analysis, and the workflow entry point (``hypo_eval``).
Registered with the core via :mod:`register`. Generic forecast-visualization
helpers remain in the core ``src/visualization`` package.
"""
