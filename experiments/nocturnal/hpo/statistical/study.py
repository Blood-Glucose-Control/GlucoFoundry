# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""statistical x Optuna sweep worker (nocturnal experiment).

Supplies the Statistical search space to the shared HPO engine with the
model-agnostic nocturnal objective. Optimizes WQL on the midnight-anchored
holdout episodes; RMSE/Brier@3.9/MACE are logged per trial.

Usage:
    python -m experiments.nocturnal.hpo.statistical.study create --config <yaml>
    python -m experiments.nocturnal.hpo.statistical.study run --n-trials 20
"""

from __future__ import annotations

from typing import Optional, Sequence

from experiments.nocturnal.hpo._shared.study_base import (
    base_default_config,
    make_study,
)
from src.workflows.hpo.engine import run_main

from .search_space import StatisticalSearchSpaceBounds

MODEL_TYPE = "statistical"

DEFAULT_CONFIG = base_default_config(
    study_name="statistical",
)

SPEC = make_study(
    MODEL_TYPE, DEFAULT_CONFIG, StatisticalSearchSpaceBounds, "Statistical"
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_main(SPEC, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
