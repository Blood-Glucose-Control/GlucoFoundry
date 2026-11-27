# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Search space for the TSMixer (Darts) nocturnal sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import optuna

from experiments.nocturnal.hpo._shared.covariates import (
    suggest_covariate_cols,
)


@dataclass
class TSMixerSearchSpaceBounds:
    """Tunable bounds for the TSMixer search space."""

    context_min: int = 128
    context_max: int = 512
    context_step: int = 128
    hidden_size_choices: Tuple[int, ...] = (32, 64, 128)
    num_blocks_min: int = 1
    num_blocks_max: int = 4
    dropout_min: float = 0.0
    dropout_max: float = 0.3
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    batch_size_choices: Tuple[int, ...] = (32, 64)
    num_epochs_choices: Tuple[int, ...] = (20, 40)

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest one TSMixer configuration for a trial."""
        # ff_size is tied to hidden_size to keep the mixer blocks balanced.
        hidden_size = trial.suggest_categorical(
            "hidden_size", list(self.hidden_size_choices)
        )
        point: Dict[str, Any] = {
            "context_length": trial.suggest_int(
                "context_length",
                self.context_min,
                self.context_max,
                step=self.context_step,
            ),
            "hidden_size": hidden_size,
            "ff_size": hidden_size,
            "num_blocks": trial.suggest_int(
                "num_blocks", self.num_blocks_min, self.num_blocks_max
            ),
            "dropout": trial.suggest_float(
                "dropout", self.dropout_min, self.dropout_max
            ),
            "learning_rate": trial.suggest_float(
                "learning_rate", self.lr_min, self.lr_max, log=True
            ),
            "batch_size": trial.suggest_categorical(
                "batch_size", list(self.batch_size_choices)
            ),
            "num_epochs": trial.suggest_categorical(
                "num_epochs", list(self.num_epochs_choices)
            ),
        }
        point["covariate_cols"] = suggest_covariate_cols(trial)
        return point
