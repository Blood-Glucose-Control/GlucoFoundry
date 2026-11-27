# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Search space for the DeepAR nocturnal sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import optuna

from experiments.nocturnal.hpo._shared.covariates import (
    suggest_covariate_cols,
)


@dataclass
class DeepARSearchSpaceBounds:
    """Tunable bounds for the DeepAR search space."""

    context_min: int = 128
    context_max: int = 512
    context_step: int = 128
    hidden_size_choices: Tuple[int, ...] = (32, 64, 128)
    num_layers_min: int = 1
    num_layers_max: int = 3
    dropout_min: float = 0.0
    dropout_max: float = 0.3
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    max_epochs_choices: Tuple[int, ...] = (30, 50)
    batch_size_choices: Tuple[int, ...] = (64, 128)

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest one DeepAR configuration for a trial."""
        point = {
            "context_length": trial.suggest_int(
                "context_length",
                self.context_min,
                self.context_max,
                step=self.context_step,
            ),
            "hidden_size": trial.suggest_categorical(
                "hidden_size", list(self.hidden_size_choices)
            ),
            "num_layers": trial.suggest_int(
                "num_layers", self.num_layers_min, self.num_layers_max
            ),
            "dropout_rate": trial.suggest_float(
                "dropout_rate", self.dropout_min, self.dropout_max
            ),
            "lr": trial.suggest_float("lr", self.lr_min, self.lr_max, log=True),
            "max_epochs": trial.suggest_categorical(
                "max_epochs", list(self.max_epochs_choices)
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", list(self.batch_size_choices)
            ),
        }
        point["covariate_cols"] = suggest_covariate_cols(trial)
        return point
