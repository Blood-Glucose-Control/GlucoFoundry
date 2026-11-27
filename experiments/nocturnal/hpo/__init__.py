# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Nocturnal (midnight-anchored) Optuna HPO sweeps, organized per model family.

Each ``<family>/`` package supplies a thin :class:`~src.workflows.hpo.engine.StudySpec`
(search space + study wiring) to the shared, experiment-agnostic HPO engine in
:mod:`src.workflows.hpo.engine`. Because nocturnal evaluation
(:func:`experiments.nocturnal.eval.evaluate_nocturnal_forecasting`) is
model-agnostic, the per-trial objective is shared in :mod:`._shared.objective`;
only the search space differs per family.

Storage: one Optuna storage (journal DB) holds the whole nocturnal experiment,
with one ``study_name`` per model family (the idiomatic Optuna layout). Each
family's worker runs under its own model interpreter (``.venvs/<family>``) but
writes to the shared study DB.
"""
