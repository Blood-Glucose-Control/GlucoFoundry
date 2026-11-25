# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Artifact-path helpers for the MetaboNet-2026 Optuna sweeps.

Per RFC-001, each experiment's run artifacts live inside ``experiments/<name>/``
(redirectable to scratch via ``GLUCOFOUNDRY_ARTIFACTS_ROOT``), not the shared
repo-root ``results/`` -- so the experiment stays self-contained and its
per-family journals never collide with another experiment's. Mirrors the
``experiments/nocturnal/hpo/_shared`` storage resolution.
"""

from __future__ import annotations

from src.experiments.registry import get_experiment_paths

EXPERIMENT = "metabonet_2026"


def family_storage(family: str) -> str:
    """Per-family Optuna journal DB under the experiment's own ``results/optuna/``."""
    results = get_experiment_paths(EXPERIMENT).results
    return str(results / "optuna" / f"{family}_metabonet_poc.journal")


def family_output_root(family: str) -> str:
    """Per-family trial-output dir under the experiment's ``trained_models/``."""
    trained = get_experiment_paths(EXPERIMENT).trained_models
    return str(trained / f"{family}_metabonet_poc_models")
