# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""deepar x Optuna sweep worker.

Supplies the deepar search space, objective, and dashboard metadata to the shared
sweep engine (:mod:`src.workflows.sweeps.engine`).

Usage:
    python -m src.workflows.sweeps.optuna_deepar.study create --config <yaml>
    python -m src.workflows.sweeps.optuna_deepar.study run --config <yaml> \\
        --n-trials 200 --gpu-label 0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import optuna

from ..engine import StudySpec, _parse_objectives, run_main
from ..optuna_chronos2.data import build_data
from .objective import DeepARObjective
from .search_space import DeepARSearchSpaceBounds

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "deepar_metabonet_poc",
    "storage": "sqlite:///results/optuna/deepar_metabonet_poc.db",
    "processed_path": "cache/data/metabonet/processed",
    "output_root": "results/optuna/deepar_metabonet_poc_models",
    "primary_metric": "mard_avg",
    "direction": "minimize",
    "time_limit": None,
    "timeout": None,
    "keep_models": False,
    "report_intermediate": False,
    "enable_pruning": False,
    "pruner": {},
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
            "hidden_size_choices": [20],
            "num_layers_choices": [1],
        },
    },
}


def build_bounds(config: Dict[str, Any]) -> DeepARSearchSpaceBounds:
    return DeepARSearchSpaceBounds(**config.get("search_space", {}))


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    """Attach a compact human-readable summary of the DeepAR search space."""
    metric_names, directions = _parse_objectives(config)
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001 - older storages may not support it
        pass

    bounds = build_bounds(config)
    data = config.get("data", {})
    objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
    study.set_user_attr("model", "DeepAR")
    study.set_user_attr("objective", f"{objective_desc} on internal temporal split")
    study.set_user_attr(
        "forecast_shape",
        f"context<= {data.get('max_context', 288)} steps -> "
        f"{data.get('forecast_length', 24)}-step forecast (+120 min); MetaboNet shape",
    )
    study.set_user_attr(
        "validation",
        f"held-out last {data.get('val_days', 2)} days/patient; "
        f"patient_limit={data.get('patient_limit')}",
    )
    study.set_user_attr(
        "logged_metrics", "rmse/mard/dts at 30/60/90/120 stored as trial user attrs"
    )
    study.set_user_attr(
        "space__context_length",
        f"int[{bounds.context_min}, {bounds.context_max}] step {bounds.context_step}",
    )
    study.set_user_attr("space__num_layers", str(bounds.num_layers_choices))
    study.set_user_attr("space__hidden_size", str(bounds.hidden_size_choices))
    study.set_user_attr(
        "space__dropout",
        f"uniform[{bounds.dropout_rate_min}, {bounds.dropout_rate_max}]",
    )
    study.set_user_attr("space__lr", f"loguniform[{bounds.lr_min}, {bounds.lr_max}]")
    study.set_user_attr(
        "space__weight_decay",
        f"loguniform[{bounds.weight_decay_min}, {bounds.weight_decay_max}]",
    )
    study.set_user_attr("space__max_epochs", str(bounds.max_epochs_choices))
    study.set_user_attr("space__batch_size", str(bounds.batch_size_choices))
    study.set_user_attr(
        "space__num_batches_per_epoch", str(bounds.num_batches_per_epoch_choices)
    )
    study.set_user_attr(
        "space__num_parallel_samples", str(bounds.num_parallel_samples_choices)
    )
    study.set_user_attr("space__target_scaler", str(bounds.target_scaler_choices))
    study.set_user_attr("space__distr_output", str(bounds.distr_output_choices))
    study.set_user_attr(
        "space__eval_metric", f"fixed={bounds.eval_metric} (not searched)"
    )
    study.set_user_attr(
        "space__known_covariates",
        "hour_sin, hour_cos (on/off together); insulin as known-future "
        f"(per-channel use_known__*): {bounds.known_candidates}",
    )
    study.set_user_attr("space__static_covariates", str(bounds.static_candidates))
    study.set_user_attr("space__insulin_as_known", str(bounds.insulin_as_known))


def build_objective(
    config: Dict[str, Any],
    data: Any,
    bounds: Any,
    gpu_label: str,
    metric_names: Sequence[str],
    is_multi: bool,
) -> Any:
    return DeepARObjective(
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
    return run_main(SPEC, argv)


if __name__ == "__main__":
    raise SystemExit(main())
