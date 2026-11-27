# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""moirai2 x Optuna sweep worker.

Supplies the moirai2 search space, objective, and dashboard metadata to the shared
sweep engine (:mod:`src.workflows.hpo.engine`).

Usage:
    python -m experiments.metabonet_2026.hpo.moirai2.study create --config <yaml>
    python -m experiments.metabonet_2026.hpo.moirai2.study run --config <yaml> \\
        --n-trials 200 --gpu-label 0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import optuna

from src.workflows.hpo.engine import StudySpec, _parse_objectives, run_main

from .._shared.data import build_data
from .._shared.paths import family_output_root, family_storage
from .objective import Moirai2Objective
from .search_space import Moirai2SearchSpaceBounds

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "moirai2_metabonet_poc",
    "storage": family_storage("moirai2"),
    "processed_path": "cache/data/metabonet/processed",
    "output_root": family_output_root("moirai2"),
    "primary_metric": "mard_avg",
    "direction": "minimize",
    "time_limit": None,
    "timeout": None,
    "keep_models": False,
    "report_intermediate": True,
    "enable_pruning": True,
    "pruner": {"type": "median", "n_startup_trials": 8, "n_warmup_steps": 1},
    "early_stopping_patience": None,
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
        "patient_limit": 100,
        "source_filter": ["iobp2"],
        "seed": 42,
    },
    "fine_tune": {
        "train_tail_days": 21,
        "val_windows_per_patient": 4,
        "max_val_windows_total": 2000,
        "val_stride_steps": 6,
        "seed": 42,
    },
    "search_space": {},
    "smoke": {
        "data": {"patient_limit": 6, "min_context": 96, "max_context": 96},
        "fine_tune": {
            "train_tail_days": 10,
            "val_windows_per_patient": 2,
            "max_val_windows_total": 16,
            "val_stride_steps": 6,
        },
        "search_space": {
            "context_min": 96,
            "context_max": 96,
            "context_step": 96,
            "lora_r_choices": [8],
            "lora_alpha_choices": [16],
            "lora_target_choices": ["attn"],
            "num_epochs_choices": [1],
            "batch_size_choices": [16],
            "beta2_choices": [0.999],
            "gradient_clip_val_choices": [1.0],
            "known_kinetics_candidates": [],
        },
    },
}


def build_bounds(config: Dict[str, Any]) -> Moirai2SearchSpaceBounds:
    return Moirai2SearchSpaceBounds(**config.get("search_space", {}))


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    metric_names, directions = _parse_objectives(config)
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001
        pass
    bounds = build_bounds(config)
    objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
    study.set_user_attr("model", f"{bounds.variant} (PEFT-LoRA fine-tune)")
    study.set_user_attr("objective", f"{objective_desc} on internal temporal split")
    purpose = config.get("purpose")
    if purpose:
        study.set_user_attr("purpose", str(purpose))
    study.set_user_attr("space__variant", bounds.variant)
    study.set_user_attr("space__model_path", str(bounds.model_path_choices))
    study.set_user_attr(
        "space__lora",
        f"r={bounds.lora_r_choices} alpha={bounds.lora_alpha_choices} "
        f"dropout[{bounds.lora_dropout_min},{bounds.lora_dropout_max}] "
        f"targets={bounds.lora_target_choices}",
    )
    study.set_user_attr("space__lr", f"loguniform[{bounds.lr_min}, {bounds.lr_max}]")
    study.set_user_attr("space__num_epochs", str(bounds.num_epochs_choices))
    study.set_user_attr("space__batch_size", str(bounds.batch_size_choices))
    study.set_user_attr(
        "space__known_covariates",
        f"always: {bounds.always_known_candidates}; "
        f"toggled kinetics: {bounds.known_kinetics_candidates}; demographics excluded",
    )


def build_objective(
    config: Dict[str, Any],
    data: Any,
    bounds: Any,
    gpu_label: str,
    metric_names: Sequence[str],
    is_multi: bool,
) -> Any:
    fine_tune = dict(config.get("fine_tune", {}))
    return Moirai2Objective(
        data,
        config["output_root"],
        bounds=bounds,  # type: ignore[arg-type]  # Moirai2 bounds used via _suggest
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
