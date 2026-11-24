# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Search space for the Statistical nocturnal sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import optuna


@dataclass
class StatisticalSearchSpaceBounds:
    """Tunable bounds for the Statistical search space."""

    model_name_choices: Tuple[str, ...] = (
        "AutoARIMA",
        "AutoETS",
        "Theta",
        "SeasonalNaive",
    )
    context_min: int = 128
    context_max: int = 512
    context_step: int = 128

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest one Statistical configuration for a trial."""
        return {
            "model_name": trial.suggest_categorical(
                "model_name", list(self.model_name_choices)
            ),
            "context_length": trial.suggest_int(
                "context_length",
                self.context_min,
                self.context_max,
                step=self.context_step,
            ),
            "covariate_cols": [],
        }
