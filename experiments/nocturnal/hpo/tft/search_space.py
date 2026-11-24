# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Search space for the TFT nocturnal sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import optuna

from experiments.nocturnal.hpo._shared.covariates import (
    suggest_covariate_cols,
)


@dataclass
class TFTSearchSpaceBounds:
    """Tunable bounds for the TFT search space."""

    context_min: int = 128
    context_max: int = 512
    context_step: int = 128
    hidden_dim_choices: Tuple[int, ...] = (16, 32, 64)
    variable_dim_choices: Tuple[int, ...] = (16, 32)
    num_heads_choices: Tuple[int, ...] = (2, 4, 8)
    dropout_min: float = 0.0
    dropout_max: float = 0.3
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    max_epochs_choices: Tuple[int, ...] = (30, 50)
    batch_size_choices: Tuple[int, ...] = (128, 256)

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest one TFT configuration for a trial."""
        point = {
            "context_length": trial.suggest_int(
                "context_length",
                self.context_min,
                self.context_max,
                step=self.context_step,
            ),
            "hidden_dim": trial.suggest_categorical(
                "hidden_dim", list(self.hidden_dim_choices)
            ),
            "variable_dim": trial.suggest_categorical(
                "variable_dim", list(self.variable_dim_choices)
            ),
            "num_heads": trial.suggest_categorical(
                "num_heads", list(self.num_heads_choices)
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
