# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""moirai x Optuna sweep worker.

Supplies the moirai search space, objective, and dashboard metadata to the shared
sweep engine (:mod:`src.workflows.sweeps.engine`).

Usage:
    python -m src.workflows.sweeps.optuna_moirai.study create --config <yaml>
    python -m src.workflows.sweeps.optuna_moirai.study run --config <yaml> \\
        --n-trials 200 --gpu-label 0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import optuna

from ..engine import StudySpec, _parse_objectives, run_main
from ..optuna_chronos2.data import build_data
from .objective import MoiraiObjective
from .search_space import MoiraiSearchSpaceBounds

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "moirai_metabonet_poc",
    "storage": "sqlite:///results/optuna/moirai_metabonet_poc.db",
    "processed_path": "cache/data/metabonet/processed",
    "output_root": "results/optuna/moirai_metabonet_poc_models",
    "primary_metric": "mard_avg",
    "direction": "minimize",
    "time_limit": None,
    "timeout": None,
    "keep_models": False,
    "report_intermediate": False,
    # Median pruning is unavailable (Moirai fine-tune does not report
    # intermediate val loss to Optuna), so it is disabled here.
    "enable_pruning": False,
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
        "patient_limit": 100,
        "source_filter": ["iobp2"],
        "seed": 42,
    },
    # Moirai fine-tune / validation-windowing knobs (per trial).
    "fine_tune": {
        "train_tail_days": 21,
        "val_windows_per_patient": 4,
        "max_val_windows_total": 2000,
        "val_stride_steps": 6,
        "seed": 42,
    },
    "search_space": {},
    "smoke": {
        "data": {
            "patient_limit": 6,
            "min_context": 96,
            "max_context": 96,
        },
        "fine_tune": {
            "train_tail_days": 10,
            "val_windows_per_patient": 2,
            "max_val_windows_total": 16,
            "val_stride_steps": 6,
        },
        "search_space": {
            "model_path_choices": ["Salesforce/moirai-1.0-R-small"],
            "context_min": 96,
            "context_max": 96,
            "context_step": 96,
            "patch_size_choices": [16],
            "finetune_pattern_choices": ["full"],
            "num_epochs_choices": [1],
            "batch_size_choices": [32],
            "beta2_choices": [0.98],
            "max_mask_ratio_choices": [0.5],
            "warmup_steps_choices": [0],
            "gradient_clip_val_choices": [1.0],
            "always_known_candidates": ["total_insulin"],
            "past_candidates": ["iob_exp"],
        },
    },
}


def build_bounds(config: Dict[str, Any]) -> MoiraiSearchSpaceBounds:
    return MoiraiSearchSpaceBounds(**config.get("search_space", {}))


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    """Attach a compact human-readable summary of the Moirai search space."""
    metric_names, directions = _parse_objectives(config)
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001 - older storages may not support it
        pass

    bounds = build_bounds(config)
    data = config.get("data", {})
    ft = config.get("fine_tune", {})
    objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
    study.set_user_attr("model", "Moirai-1.x (native fine-tune)")
    study.set_user_attr("objective", f"{objective_desc} on internal temporal split")
    purpose = config.get("purpose")
    if purpose:
        study.set_user_attr("purpose", str(purpose))
    study.set_user_attr(
        "forecast_shape",
        f"context in [{bounds.context_min}, {bounds.context_max}] steps -> "
        f"{bounds.forecast_length}-step forecast (+120 min); insulin as "
        "known-future covariate spanning context + horizon",
    )
    study.set_user_attr(
        "validation",
        f"held-out last {data.get('val_days', 2)} days/patient; "
        f"patient_limit={data.get('patient_limit')}; source={data.get('source_filter')}; "
        f"train_tail_days={ft.get('train_tail_days')}; "
        f"val_windows_per_patient={ft.get('val_windows_per_patient')}",
    )
    study.set_user_attr(
        "logged_metrics", "rmse/mard/dts at 30/60/90/120 stored as trial user attrs"
    )
    study.set_user_attr("space__model_path", str(bounds.model_path_choices))
    study.set_user_attr(
        "space__context_length",
        f"int[{bounds.context_min}, {bounds.context_max}] step {bounds.context_step}",
    )
    study.set_user_attr("space__patch_size", str(bounds.patch_size_choices))
    study.set_user_attr("space__lr", f"loguniform[{bounds.lr_min}, {bounds.lr_max}]")
    study.set_user_attr(
        "space__weight_decay",
        f"loguniform[{bounds.weight_decay_min}, {bounds.weight_decay_max}]",
    )
    study.set_user_attr("space__finetune_pattern", str(bounds.finetune_pattern_choices))
    study.set_user_attr("space__num_epochs", str(bounds.num_epochs_choices))
    study.set_user_attr("space__batch_size", str(bounds.batch_size_choices))
    study.set_user_attr(
        "space__betas", f"beta1={bounds.beta1} (fixed); beta2={bounds.beta2_choices}"
    )
    study.set_user_attr(
        "space__mask_ratio",
        f"min={bounds.min_mask_ratio} (fixed); max={bounds.max_mask_ratio_choices}",
    )
    study.set_user_attr("space__warmup_steps", str(bounds.warmup_steps_choices))
    study.set_user_attr(
        "space__gradient_clip_val", str(bounds.gradient_clip_val_choices)
    )
    study.set_user_attr(
        "space__known_covariates",
        f"always known-future insulin: {bounds.always_known_candidates}; "
        f"always known-future kinetics: {bounds.always_known_kinetics}; "
        f"toggled known kinetics: {bounds.known_kinetics_candidates}; "
        f"sex-known={bounds.include_sex_known}; "
        f"demographic bundle (use_demographics): {bounds.demographic_bundle}",
    )
    study.set_user_attr("space__past_covariates", str(bounds.past_candidates))


def build_objective(
    config: Dict[str, Any],
    data: Any,
    bounds: Any,
    gpu_label: str,
    metric_names: Sequence[str],
    is_multi: bool,
) -> Any:
    fine_tune = dict(config.get("fine_tune", {}))
    return MoiraiObjective(
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
    return run_main(SPEC, argv)


if __name__ == "__main__":
    raise SystemExit(main())
