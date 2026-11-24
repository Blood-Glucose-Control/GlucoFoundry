# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""timesfm x Optuna sweep worker (nocturnal experiment).

Supplies the TimesFM search space to the shared HPO engine with the
model-agnostic nocturnal objective. Optimizes WQL on the midnight-anchored
holdout episodes; RMSE/Brier@3.9/MACE are logged per trial.

Usage:
    python -m experiments.nocturnal.hpo.timesfm.study create --config <yaml>
    python -m experiments.nocturnal.hpo.timesfm.study run --n-trials 20
"""

from __future__ import annotations

from typing import Optional, Sequence

from experiments.nocturnal.hpo._shared.study_base import (
    base_default_config,
    make_study,
)
from src.workflows.hpo.engine import run_main

from .search_space import TimesFMSearchSpaceBounds

MODEL_TYPE = "timesfm"

DEFAULT_CONFIG = base_default_config(
    study_name="timesfm",
    output_root="results/optuna/nocturnal_timesfm_models",
)

SPEC = make_study(MODEL_TYPE, DEFAULT_CONFIG, TimesFMSearchSpaceBounds, "TimesFM")


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_main(SPEC, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
