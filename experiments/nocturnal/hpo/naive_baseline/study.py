# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""naive_baseline x Optuna sweep worker (nocturnal experiment).

Supplies the naive-baseline search space to the shared HPO engine
(:mod:`src.workflows.hpo.engine`) with the model-agnostic nocturnal objective
(:mod:`experiments.nocturnal.hpo._shared.objective`). Optimizes WQL on the
midnight-anchored holdout episodes; RMSE/Brier@3.9/MACE are logged per trial.

Usage:
    python -m experiments.nocturnal.hpo.naive_baseline.study create --config <yaml>
    python -m experiments.nocturnal.hpo.naive_baseline.study run --config <yaml> \\
        --n-trials 20
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import optuna

from experiments.nocturnal.hpo._shared.data import build_data as _build_data
from experiments.nocturnal.hpo._shared.objective import (
    build_objective as _build_objective,
)
from src.workflows.hpo.engine import StudySpec, _parse_objectives, run_main

from .search_space import NaiveSearchSpaceBounds

logger = logging.getLogger(__name__)

MODEL_TYPE = "naive_baseline"

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "naive_baseline",
    # One shared journal DB for the whole nocturnal experiment; study_name
    # distinguishes families (the idiomatic Optuna multi-study layout).
    "storage": "results/optuna/nocturnal_hpo.journal",
    "storage_backend": "journal",
    "output_root": "results/optuna/nocturnal_naive_baseline_models",
    "primary_metric": "wql",
    "direction": "minimize",
    "timeout": None,
    "keep_models": False,
    "compute_dilate": False,
    "early_stopping_patience": None,
    "data": {
        "datasets": ["aleppo_2017", "brown_2019", "lynch_2022"],
        "holdout_config_dir": "configs/data/holdout_10pct",
        "forecast_length": 96,
        "patient_limit": None,
    },
    "search_space": {},
    "smoke": {
        "data": {
            "datasets": ["aleppo_2017"],
            "patient_limit": 2,
        },
        "search_space": {
            "context_min": 128,
            "context_max": 256,
            "context_step": 128,
        },
    },
}


def build_bounds(config: Dict[str, Any]) -> NaiveSearchSpaceBounds:
    return NaiveSearchSpaceBounds(**config.get("search_space", {}))


def build_objective(
    config: Dict[str, Any],
    data: Any,
    bounds: Any,
    gpu_label: str,
    metric_names: Sequence[str],
    is_multi: bool,
) -> Any:
    return _build_objective(
        config, data, bounds, gpu_label, metric_names, is_multi, model_type=MODEL_TYPE
    )


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    metric_names, directions = _parse_objectives(config)
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001 - older storages may not support it
        pass
    bounds = build_bounds(config)
    data = config.get("data", {})
    objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
    study.set_user_attr("model", "NaiveBaseline")
    study.set_user_attr(
        "objective", f"{objective_desc} on midnight-anchored nocturnal holdout"
    )
    study.set_user_attr("datasets", ", ".join(data.get("datasets", [])))
    study.set_user_attr(
        "forecast_shape",
        f"context<= {bounds.context_max} steps -> "
        f"{data.get('forecast_length', 96)}-step overnight forecast",
    )
    study.set_user_attr("space__model_name", str(bounds.model_name_choices))
    study.set_user_attr(
        "space__context_length",
        f"int[{bounds.context_min}, {bounds.context_max}] step {bounds.context_step}",
    )
    study.set_user_attr(
        "logged_metrics", "rmse/wql/brier/mace stored as trial user attrs"
    )


SPEC = StudySpec(
    default_config=DEFAULT_CONFIG,
    build_bounds=build_bounds,
    build_objective=build_objective,
    build_data=_build_data,
    apply_dashboard_metadata=_apply_dashboard_metadata,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_main(SPEC, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
