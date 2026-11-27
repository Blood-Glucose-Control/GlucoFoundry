# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Shared covariate-set selection for nocturnal sweeps.

The manual ablation grids mainly varied which covariates a model sees (BG-only
vs BG+IOB vs BG+IOB+COB, etc.). This exposes those as a single categorical search
dimension so covariate-capable families can tune it uniformly. Only columns that
exist for every dataset are used.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import optuna

# Named covariate sets -> concrete columns (all present post-standardization).
COVARIATE_SETS: Dict[str, List[str]] = {
    "none": [],
    "iob": ["iob"],
    "iob_insulin_availability": ["iob", "insulin_availability"],
    "iob_cob": ["iob", "cob"],
}

DEFAULT_COVARIATE_CHOICES: Sequence[str] = ("none", "iob", "iob_cob")


def suggest_covariate_cols(
    trial: optuna.Trial, choices: Sequence[str] = DEFAULT_COVARIATE_CHOICES
) -> List[str]:
    """Suggest a covariate column set for a trial (returns the concrete columns)."""
    name = trial.suggest_categorical("covariate_set", list(choices))
    return list(COVARIATE_SETS[name])
