# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Search space for the Chronos-2 nocturnal sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import optuna

from experiments.nocturnal.hpo._shared.covariates import (
    suggest_covariate_cols,
)


@dataclass
class Chronos2SearchSpaceBounds:
    """Tunable bounds for the Chronos-2 search space."""

    model_path: str = "autogluon/chronos-2"
    context_min: int = 128
    context_max: int = 512
    context_step: int = 128
    fine_tune_lr_min: float = 1e-6
    fine_tune_lr_max: float = 1e-4
    fine_tune_steps_choices: Tuple[int, ...] = (1000, 5000, 10000)
    fine_tune_batch_size_choices: Tuple[int, ...] = (128, 256)

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest one Chronos-2 configuration for a trial."""
        point = {
            "model_path": self.model_path,
            "training_mode": "fine_tune",
            "context_length": trial.suggest_int(
                "context_length",
                self.context_min,
                self.context_max,
                step=self.context_step,
            ),
            "fine_tune_lr": trial.suggest_float(
                "fine_tune_lr", self.fine_tune_lr_min, self.fine_tune_lr_max, log=True
            ),
            "fine_tune_steps": trial.suggest_categorical(
                "fine_tune_steps", list(self.fine_tune_steps_choices)
            ),
            "fine_tune_batch_size": trial.suggest_categorical(
                "fine_tune_batch_size", list(self.fine_tune_batch_size_choices)
            ),
        }
        point["covariate_cols"] = suggest_covariate_cols(trial)
        return point
