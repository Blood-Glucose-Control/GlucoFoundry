# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Search space for the GBM (AutoGluon tabular) nocturnal sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import optuna

from experiments.nocturnal.hpo._shared.covariates import (
    suggest_covariate_cols,
)


@dataclass
class GBMSearchSpaceBounds:
    """Tunable bounds for the GBM (gradient-boosting tabular) search space."""

    context_min: int = 144
    context_max: int = 288
    context_step: int = 48
    strategy_choices: Tuple[str, ...] = ("recursive", "direct")
    target_scaler_choices: Tuple[Optional[str], ...] = (None, "standard", "robust")
    num_boost_round_min: int = 100
    num_boost_round_max: int = 500
    num_boost_round_step: int = 100
    lr_min: float = 1e-2
    lr_max: float = 2e-1
    # Lag windows (indexed categorically; Optuna cannot suggest a list directly).
    lags_choices: Tuple[Tuple[int, ...], ...] = (
        (1, 2, 3, 6, 12),
        (1, 2, 3, 6, 12, 24, 36),
    )

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest one GBM configuration for a trial."""
        lags_idx = trial.suggest_categorical(
            "lags_idx", list(range(len(self.lags_choices)))
        )
        point: Dict[str, Any] = {
            "context_length": trial.suggest_int(
                "context_length",
                self.context_min,
                self.context_max,
                step=self.context_step,
            ),
            "strategy": trial.suggest_categorical(
                "strategy", list(self.strategy_choices)
            ),
            "target_scaler": trial.suggest_categorical(
                "target_scaler", list(self.target_scaler_choices)
            ),
            "lags": list(self.lags_choices[lags_idx]),
            "model_hyperparameters": {
                "num_boost_round": trial.suggest_int(
                    "num_boost_round",
                    self.num_boost_round_min,
                    self.num_boost_round_max,
                    step=self.num_boost_round_step,
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate", self.lr_min, self.lr_max, log=True
                ),
            },
        }
        point["covariate_cols"] = suggest_covariate_cols(trial)
        return point
