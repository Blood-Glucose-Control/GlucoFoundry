# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""naive_baseline x Optuna sweep worker (nocturnal experiment).

Supplies the naive-baseline search space to the shared HPO engine with the
model-agnostic nocturnal objective. Optimizes WQL on the midnight-anchored
holdout episodes; RMSE/Brier@3.9/MACE are logged per trial. This family is the
absolute performance floor and a fast end-to-end smoke of the nocturnal engine.

Usage:
    python -m experiments.nocturnal.hpo.naive_baseline.study create --config <yaml>
    python -m experiments.nocturnal.hpo.naive_baseline.study run --n-trials 20
"""

from __future__ import annotations

from typing import Optional, Sequence

from experiments.nocturnal.hpo._shared.study_base import (
    base_default_config,
    make_study,
)
from src.workflows.hpo.engine import run_main

from .search_space import NaiveSearchSpaceBounds

MODEL_TYPE = "naive_baseline"

DEFAULT_CONFIG = base_default_config(
    study_name="naive_baseline",
    output_root="results/optuna/nocturnal_naive_baseline_models",
    smoke_search_space={"context_min": 128, "context_max": 256, "context_step": 128},
)

SPEC = make_study(MODEL_TYPE, DEFAULT_CONFIG, NaiveSearchSpaceBounds, "NaiveBaseline")


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_main(SPEC, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
