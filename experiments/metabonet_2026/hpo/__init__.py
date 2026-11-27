# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""MetaboNet-2026 Optuna HPO sweeps, organized per model family.

Each ``<family>/`` package supplies a thin :class:`~src.workflows.hpo.engine.StudySpec`
(search space, objective, data builder, dashboard metadata) to the shared,
experiment-agnostic HPO engine in :mod:`src.workflows.hpo.engine`. Cross-family
MetaboNet data/metrics/features/kinetics live in :mod:`._shared`.
"""
