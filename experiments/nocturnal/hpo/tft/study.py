# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""tft x Optuna sweep worker (nocturnal experiment).

Supplies the TFT search space to the shared HPO engine with the
model-agnostic nocturnal objective. Optimizes WQL on the midnight-anchored
holdout episodes; RMSE/Brier@3.9/MACE are logged per trial.

Usage:
    python -m experiments.nocturnal.hpo.tft.study create --config <yaml>
    python -m experiments.nocturnal.hpo.tft.study run --n-trials 20
"""

from __future__ import annotations

from typing import Optional, Sequence

from experiments.nocturnal.hpo._shared.study_base import (
    base_default_config,
    make_study,
)
from src.workflows.hpo.engine import run_main

from .search_space import TFTSearchSpaceBounds

MODEL_TYPE = "tft"

DEFAULT_CONFIG = base_default_config(
    study_name="tft",
    output_root="results/optuna/nocturnal_tft_models",
)

SPEC = make_study(MODEL_TYPE, DEFAULT_CONFIG, TFTSearchSpaceBounds, "TFT")


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_main(SPEC, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
