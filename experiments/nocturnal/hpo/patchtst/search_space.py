# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Search space for the PatchTST nocturnal sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import optuna

from experiments.nocturnal.hpo._shared.covariates import (
    suggest_covariate_cols,
)


@dataclass
class PatchTSTSearchSpaceBounds:
    """Tunable bounds for the PatchTST search space."""

    context_min: int = 128
    context_max: int = 512
    context_step: int = 128
    patch_len_choices: Tuple[int, ...] = (8, 16)
    stride_choices: Tuple[int, ...] = (4, 8)
    d_model_choices: Tuple[int, ...] = (64, 128, 256)
    nhead_choices: Tuple[int, ...] = (8, 16)
    num_encoder_layers_min: int = 2
    num_encoder_layers_max: int = 4
    lr_min: float = 1e-5
    lr_max: float = 1e-3
    max_epochs_choices: Tuple[int, ...] = (30, 50)
    batch_size_choices: Tuple[int, ...] = (128, 256)

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest one PatchTST configuration for a trial."""
        point = {
            "context_length": trial.suggest_int(
                "context_length",
                self.context_min,
                self.context_max,
                step=self.context_step,
            ),
            "patch_len": trial.suggest_categorical(
                "patch_len", list(self.patch_len_choices)
            ),
            "stride": trial.suggest_categorical("stride", list(self.stride_choices)),
            "d_model": trial.suggest_categorical("d_model", list(self.d_model_choices)),
            "nhead": trial.suggest_categorical("nhead", list(self.nhead_choices)),
            "num_encoder_layers": trial.suggest_int(
                "num_encoder_layers",
                self.num_encoder_layers_min,
                self.num_encoder_layers_max,
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
