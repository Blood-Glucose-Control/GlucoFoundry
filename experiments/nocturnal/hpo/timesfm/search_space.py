# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Search space for the TimesFM nocturnal sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import optuna


@dataclass
class TimesFMSearchSpaceBounds:
    """Tunable bounds for the TimesFM search space."""

    context_min: int = 128
    context_max: int = 512
    context_step: int = 128
    lr_min: float = 1e-6
    lr_max: float = 1e-4
    num_epochs_choices: Tuple[int, ...] = (20, 35)
    batch_size_choices: Tuple[int, ...] = (128, 256)

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest one TimesFM configuration for a trial."""
        return {
            "training_mode": "fine_tune",
            "context_length": trial.suggest_int(
                "context_length",
                self.context_min,
                self.context_max,
                step=self.context_step,
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", self.lr_min, self.lr_max, log=True
            ),
            "num_epochs": trial.suggest_categorical(
                "num_epochs", list(self.num_epochs_choices)
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", list(self.batch_size_choices)
            ),
            "covariate_cols": [],
        }
