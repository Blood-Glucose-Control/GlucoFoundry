# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Search space for the Moirai-2 nocturnal sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import optuna


@dataclass
class Moirai2SearchSpaceBounds:
    """Tunable bounds for the Moirai-2 search space."""

    model_path: str = "Salesforce/moirai-2.0-R-small"
    context_min: int = 128
    context_max: int = 512
    context_step: int = 128
    lora_r_choices: Tuple[int, ...] = (8, 16, 32)
    lora_alpha_choices: Tuple[int, ...] = (16, 32, 64)
    lora_dropout_min: float = 0.0
    lora_dropout_max: float = 0.1
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    num_epochs_choices: Tuple[int, ...] = (3, 5)

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest one Moirai-2 configuration for a trial."""
        return {
            "model_path": self.model_path,
            "variant": "moirai2",
            "training_mode": "fine_tune",
            "context_length": trial.suggest_int(
                "context_length",
                self.context_min,
                self.context_max,
                step=self.context_step,
            ),
            "lora_r": trial.suggest_categorical("lora_r", list(self.lora_r_choices)),
            "lora_alpha": trial.suggest_categorical(
                "lora_alpha", list(self.lora_alpha_choices)
            ),
            "lora_dropout": trial.suggest_float(
                "lora_dropout", self.lora_dropout_min, self.lora_dropout_max
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", self.lr_min, self.lr_max, log=True
            ),
            "num_epochs": trial.suggest_categorical(
                "num_epochs", list(self.num_epochs_choices)
            ),
            "covariate_cols": [],
        }
