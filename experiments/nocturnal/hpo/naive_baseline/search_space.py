# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Search space for the naive-baseline nocturnal sweep.

The naive baselines (last-value ``Naive``, historical-mean ``Average``) ignore
covariates, so the only tuned dimensions are the baseline variant and the
context window length. This family is primarily an absolute performance floor and
a fast end-to-end smoke of the nocturnal HPO engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import optuna


@dataclass
class NaiveSearchSpaceBounds:
    """Tunable bounds for the naive-baseline search space."""

    model_name_choices: Tuple[str, ...] = ("Naive", "Average")
    context_min: int = 128
    context_max: int = 512
    context_step: int = 128

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest one naive-baseline configuration for a trial."""
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
