# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""timesfm3 x Optuna sweep worker.

Supplies the timesfm3 search space, objective, and dashboard metadata to the shared
sweep engine (:mod:`src.workflows.hpo.engine`).

Usage:
    python -m experiments.metabonet_2026.hpo.timesfm3.study create --config <yaml>
    python -m experiments.metabonet_2026.hpo.timesfm3.study run --config <yaml> \\
        --n-trials 200 --gpu-label 0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import optuna

from src.workflows.hpo.engine import StudySpec, _parse_objectives, run_main

from .._shared.data import build_data
from .objective import TimesFM3Objective
from .search_space import TimesFM3SearchSpaceBounds

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "timesfm3_metabonet_poc",
    "storage": "sqlite:///results/optuna/timesfm3_metabonet_poc.db",
    "processed_path": "cache/data/metabonet/processed",
    "output_root": "results/optuna/timesfm3_metabonet_poc_models",
    "primary_metric": "mard_avg",
    "direction": "minimize",
    "time_limit": None,
    "timeout": None,
    "keep_models": False,
    "report_intermediate": True,
    "enable_pruning": True,
    "pruner": {"type": "median", "n_startup_trials": 8, "n_warmup_steps": 1},
    "early_stopping_patience": None,  # worker-level (stop worker if study stalls)
    "heartbeat_interval": 60,
    "grace_period": 600,
    "max_retry": 2,
    "data": {
        "forecast_length": 24,
        "max_context": 288,
        "min_context": 96,
        "val_days": 2,
        "interval_mins": 5,
        "imputation_threshold_mins": 30,
        "eval_stride_steps": 3,
        "max_val_windows_per_patient": 8,
        "max_val_windows_total": 2000,
        "patient_limit": None,
        "seed": 42,
    },
    # TimesFM 3.0 fine-tune / windowing knobs (per trial).
    "fine_tune": {
        "model_path": "google/timesfm-3.0-pytorch",
        "device": "cuda",
        "train_windows_per_patient": 24,
        "max_train_windows": 6000,
        "val_windows_per_patient": 4,
        "max_val_windows_total": 2000,
        "val_stride_steps": 6,
        "es_val_fraction": 0.2,
        "es_patience": 1,
        "covariate_jitter": 0.001,
        "grad_clip_norm": 1.0,
        "seed": 42,
        "adapters_dir": None,
        "keep_top_k": 0,
    },
    "search_space": {},
    "smoke": {
        "data": {
            "patient_limit": 8,
            "min_context": 288,
            "max_context": 288,
        },
        "fine_tune": {
            "train_windows_per_patient": 3,
            "max_train_windows": 40,
            "val_windows_per_patient": 2,
            "max_val_windows_total": 16,
            "val_stride_steps": 6,
            "es_val_fraction": 0.25,
            "es_patience": 1,
        },
        "search_space": {
            "context_min": 96,
            "context_max": 96,
            "context_step": 32,
            "lora_r_choices": [4],
            "lora_alpha_choices": [8],
            "lora_dropout_min": 0.0,
            "lora_dropout_max": 0.0,
            "lora_target_modules_choices": ["all-linear"],
            "use_dora_choices": [False],
            "use_rslora_choices": [False],
            "use_znorm_choices": [False],
            "lr_schedule_choices": ["constant"],
            "max_epochs_choices": [1],
            "batch_size_choices": [8],
        },
    },
}


def build_bounds(config: Dict[str, Any]) -> TimesFM3SearchSpaceBounds:
    return TimesFM3SearchSpaceBounds(**config.get("search_space", {}))


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    """Attach a compact human-readable summary of the TimesFM 3.0 search space."""
    metric_names, directions = _parse_objectives(config)
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001 - older storages may not support it
        pass

    bounds = build_bounds(config)
    data = config.get("data", {})
    ft = config.get("fine_tune", {})
    objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
    study.set_user_attr("model", "TimesFM-3.0 (LoRA fine-tune)")
    study.set_user_attr("objective", f"{objective_desc} on internal temporal split")
    purpose = config.get("purpose")
    if purpose:
        study.set_user_attr("purpose", str(purpose))
    study.set_user_attr(
        "forecast_shape",
        f"context in [{bounds.context_min}, {bounds.context_max}] steps -> "
        f"{bounds.forecast_length}-step forecast (+120 min); known-future "
        "covariates span context + horizon",
    )
    study.set_user_attr(
        "validation",
        f"held-out last {data.get('val_days', 2)} days/patient; "
        f"patient_limit={data.get('patient_limit')}; "
        f"val_windows_per_patient={ft.get('val_windows_per_patient')}",
    )
    study.set_user_attr(
        "logged_metrics", "rmse/mard/dts at 30/60/90/120 stored as trial user attrs"
    )
    study.set_user_attr(
        "space__context_length",
        f"int[{bounds.context_min}, {bounds.context_max}] step {bounds.context_step}",
    )
    study.set_user_attr("space__lora_r", str(bounds.lora_r_choices))
    study.set_user_attr("space__lora_alpha", str(bounds.lora_alpha_choices))
    study.set_user_attr(
        "space__lora_dropout",
        f"uniform[{bounds.lora_dropout_min}, {bounds.lora_dropout_max}]",
    )
    study.set_user_attr(
        "space__lora_target_modules", str(bounds.lora_target_modules_choices)
    )
    study.set_user_attr(
        "space__lora_variants",
        f"use_dora={bounds.use_dora_choices} use_rslora={bounds.use_rslora_choices}",
    )
    study.set_user_attr("space__use_znorm", str(bounds.use_znorm_choices))
    study.set_user_attr("space__lr", f"loguniform[{bounds.lr_min}, {bounds.lr_max}]")
    study.set_user_attr(
        "space__weight_decay",
        f"loguniform[{bounds.weight_decay_min}, {bounds.weight_decay_max}]",
    )
    study.set_user_attr(
        "space__lr_schedule",
        f"{bounds.lr_schedule_choices}; warmup_ratio "
        f"uniform[{bounds.warmup_ratio_min}, {bounds.warmup_ratio_max}]",
    )
    study.set_user_attr("space__max_epochs", str(bounds.max_epochs_choices))
    study.set_user_attr("space__batch_size", str(bounds.batch_size_choices))
    study.set_user_attr(
        "space__eval_metric", f"fixed={bounds.eval_metric} (not searched)"
    )
    study.set_user_attr(
        "space__known_covariates",
        "hour_sin, hour_cos (on/off together); insulin as known-future "
        f"(per-channel use_known__*): {bounds.known_candidates}; "
        f"always-on: {bounds.always_known_cols}; "
        f"sex-known={bounds.include_sex}; "
        f"demographic bundle (use_demographics): {bounds.demographic_bundle}",
    )
    study.set_user_attr("space__insulin_as_known", str(bounds.insulin_as_known))


def build_objective(
    config: Dict[str, Any],
    data: Any,
    bounds: Any,
    gpu_label: str,
    metric_names: Sequence[str],
    is_multi: bool,
) -> Any:
    fine_tune = dict(config.get("fine_tune", {}))
    return TimesFM3Objective(
        data,
        config["output_root"],
        bounds=bounds,
        time_limit=config.get("time_limit"),
        keep_models=bool(config.get("keep_models", False)),
        gpu_label=gpu_label,
        primary_metric=metric_names[0],
        objective_metrics=list(metric_names) if is_multi else None,
        report_intermediate=bool(config.get("report_intermediate", False)),
        **fine_tune,
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
