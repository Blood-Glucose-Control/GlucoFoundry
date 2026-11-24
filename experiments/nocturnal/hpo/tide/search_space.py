# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Search space for the TiDE nocturnal sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import optuna

from experiments.nocturnal.hpo._shared.covariates import (
    suggest_covariate_cols,
)


@dataclass
class TiDESearchSpaceBounds:
    """Tunable bounds for the TiDE search space."""

    context_min: int = 128
    context_max: int = 512
    context_step: int = 128
    encoder_hidden_choices: Tuple[int, ...] = (128, 256)
    decoder_hidden_choices: Tuple[int, ...] = (128, 256)
    temporal_hidden_choices: Tuple[int, ...] = (64, 128, 256)
    num_layers_min: int = 1
    num_layers_max: int = 3
    dropout_min: float = 0.0
    dropout_max: float = 0.3
    lr_min: float = 1e-4
    lr_max: float = 1e-2
    batch_size_choices: Tuple[int, ...] = (128, 256)

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest one TiDE configuration for a trial."""
        point = {
            "context_length": trial.suggest_int(
                "context_length",
                self.context_min,
                self.context_max,
                step=self.context_step,
            ),
            "encoder_hidden_dim": trial.suggest_categorical(
                "encoder_hidden_dim", list(self.encoder_hidden_choices)
            ),
            "decoder_hidden_dim": trial.suggest_categorical(
                "decoder_hidden_dim", list(self.decoder_hidden_choices)
            ),
            "temporal_hidden_dim": trial.suggest_categorical(
                "temporal_hidden_dim", list(self.temporal_hidden_choices)
            ),
            "num_layers_encoder": trial.suggest_int(
                "num_layers_encoder", self.num_layers_min, self.num_layers_max
            ),
            "num_layers_decoder": trial.suggest_int(
                "num_layers_decoder", self.num_layers_min, self.num_layers_max
            ),
            "dropout": trial.suggest_float(
                "dropout", self.dropout_min, self.dropout_max
            ),
            "lr": trial.suggest_float("lr", self.lr_min, self.lr_max, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", list(self.batch_size_choices)
            ),
        }
        point["covariate_cols"] = suggest_covariate_cols(trial)
        return point
