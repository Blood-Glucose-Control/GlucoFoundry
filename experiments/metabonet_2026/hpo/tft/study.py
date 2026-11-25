# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""tft x Optuna sweep worker.

Supplies the tft search space, objective, and dashboard metadata to the shared
sweep engine (:mod:`src.workflows.hpo.engine`).

Usage:
    python -m experiments.metabonet_2026.hpo.tft.study create --config <yaml>
    python -m experiments.metabonet_2026.hpo.tft.study run --config <yaml> \\
        --n-trials 200 --gpu-label 0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import optuna

from src.workflows.hpo.engine import StudySpec, run_main

from .._shared.data import build_data
from .._shared.paths import family_output_root, family_storage
from .objective import TFTObjective
from .search_space import TFTSearchSpaceBounds

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "tft_metabonet_poc",
    "storage": family_storage("tft"),
    "processed_path": "cache/data/metabonet/processed",
    "output_root": family_output_root("tft"),
    "primary_metric": "mard_avg",
    "direction": "minimize",
    "time_limit": None,
    "timeout": None,
    "keep_models": False,
    "early_stopping_patience": None,  # worker-level (stop worker if study stalls)
    "heartbeat_interval": 60,
    "grace_period": 300,
    "max_retry": 2,
    "data": {
        "forecast_length": 24,
        "max_context": 288,
        "min_context": 48,
        "val_days": 2,
        "interval_mins": 5,
        "imputation_threshold_mins": 30,
        "eval_stride_steps": 3,
        "max_val_windows_per_patient": 8,
        "max_val_windows_total": 2000,
        "patient_limit": None,
        "seed": 42,
    },
    "search_space": {},
    "smoke": {
        "data": {
            "patient_limit": 8,
            "max_val_windows_per_patient": 3,
            "max_val_windows_total": 18,
        },
        "search_space": {
            "context_min": 48,
            "context_max": 96,
            "context_step": 48,
            "max_epochs_choices": [2],
            "num_batches_per_epoch_choices": [5],
            "batch_size_choices": [16],
        },
    },
}


def build_bounds(config: Dict[str, Any]) -> TFTSearchSpaceBounds:
    return TFTSearchSpaceBounds(**config.get("search_space", {}))


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    """Attach a compact human-readable summary of the TFT search space."""
    bounds = build_bounds(config)
    study.set_user_attr("model", "TemporalFusionTransformer")
    study.set_user_attr(
        "space__context_length",
        f"int[{bounds.context_min}, {bounds.context_max}] step {bounds.context_step}",
    )
    study.set_user_attr("space__hidden_dim", str(bounds.hidden_dim_choices))
    study.set_user_attr("space__variable_dim", str(bounds.variable_dim_choices))
    study.set_user_attr("space__num_heads", str(bounds.num_heads_choices))
    study.set_user_attr("space__dropout", f"uniform[0, {bounds.dropout_max}]")
    study.set_user_attr("space__lr", f"loguniform[{bounds.lr_min}, {bounds.lr_max}]")
    study.set_user_attr("space__max_epochs", str(bounds.max_epochs_choices))
    study.set_user_attr("space__batch_size", str(bounds.batch_size_choices))
    study.set_user_attr(
        "space__num_batches_per_epoch", str(bounds.num_batches_per_epoch_choices)
    )
    study.set_user_attr("space__target_scaler", str(bounds.target_scaler_choices))
    study.set_user_attr("space__eval_metric", str(bounds.eval_metric_choices))


def build_objective(
    config: Dict[str, Any],
    data: Any,
    bounds: Any,
    gpu_label: str,
    metric_names: Sequence[str],
    is_multi: bool,
) -> Any:
    return TFTObjective(
        data,
        config["output_root"],
        bounds=bounds,
        time_limit=config.get("time_limit"),
        keep_models=bool(config.get("keep_models", False)),
        gpu_label=gpu_label,
        primary_metric=metric_names[0],
        objective_metrics=list(metric_names) if is_multi else None,
        report_intermediate=bool(config.get("report_intermediate", False)),
    )


SPEC = StudySpec(
    default_config=DEFAULT_CONFIG,
    build_bounds=build_bounds,
    build_objective=build_objective,
    build_data=build_data,
    apply_dashboard_metadata=_apply_dashboard_metadata,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run_main(SPEC, argv, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
