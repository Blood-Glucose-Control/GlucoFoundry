# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Chronos-2 x Optuna sweep worker.

Supplies the Chronos-2 search space, objective, and dashboard metadata to the
shared sweep engine (:mod:`src.workflows.sweeps.engine`).

Usage:
    python -m src.workflows.sweeps.optuna_chronos2.study create --config <yaml>
    python -m src.workflows.sweeps.optuna_chronos2.study run --config <yaml> \\
        --n-trials 20 --gpu-label 0

Launch cooperative workers with
``experiments/nocturnal/scripts/sweeps/optuna/launch_optuna_tmux.sh``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Dict, Optional

import optuna

from ..engine import StudySpec, _parse_objectives, run_main
from .data import build_data
from .objective import Chronos2Objective
from .search_space import SearchSpaceBounds

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "chronos2_metabonet_poc",
    "storage": "sqlite:///optuna_chronos2_poc.db",
    "direction": "minimize",
    "processed_path": "cache/data/metabonet/processed",
    "output_root": "trained_models/optuna_chronos2_poc",
    "time_limit": None,
    "keep_models": False,
    # Which logged metric Optuna minimizes ("rmse_120" default, or "mard_120").
    "primary_metric": "rmse_120",
    # Early-terminate weak trials during fine-tuning by reporting the HF
    # validation loss to Optuna every ~100 steps. `pruner` selects the strategy
    # ("median" | "asha" | "hyperband") and its parameters.
    "enable_pruning": False,
    "pruner": {},
    # Wall-clock budget per worker (seconds); None = run until N_TRIALS.
    "timeout": None,
    # Stop a worker early if the study's best value has not improved for this
    # many completed trials (0/None = disabled; keep exploring for ensembling).
    "early_stopping_patience": None,
    # HF trainer early stopping WITHIN a trial: stop fine-tuning once eval_loss
    # has not improved for this many evals (0/None = disabled). Saves compute on
    # configs that plateau well before their fine_tune_steps budget.
    "fine_tune_early_stopping_patience": None,
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
            "steps_choices": [10],
            "batch_size_choices": [8],
        },
    },
}


def build_bounds(config: Dict[str, Any]) -> SearchSpaceBounds:
    return SearchSpaceBounds(**config.get("search_space", {}))


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    """Label the objective(s) and attach study notes for the Optuna dashboard.

    Everything here is stored in the Optuna storage itself, so the VSCode Optuna
    Dashboard extension (or optuna-dashboard) shows a readable objective axis and
    a description panel without any extra tooling.
    """
    metric_names, directions = _parse_objectives(config)
    try:
        study.set_metric_names(metric_names)
    except Exception:  # pragma: no cover - older optuna
        pass

    search = config.get("search_space", {})
    data = config.get("data", {})
    objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
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

    # Full search space, so the ranges are visible in the dashboard's study
    # "User Attributes" (trials only show sampled values, not the space).
    from .search_space import (
        INSULIN_OBSERVED_COL,
        PAST_COVARIATE_MASKS,
        PIECEWISE_STATIC_CANDIDATES,
        STATIC_COVARIATE_CANDIDATES,
        SearchSpaceBounds,
    )

    bounds = SearchSpaceBounds(**search) if search else SearchSpaceBounds()
    study.set_user_attr(
        "space__context_length",
        f"int[{bounds.context_min}..{bounds.context_max}] step {bounds.context_step}",
    )
    study.set_user_attr(
        "space__fine_tune_lr", f"loguniform[{bounds.lr_min}, {bounds.lr_max}]"
    )
    study.set_user_attr("space__fine_tune_steps", str(bounds.steps_choices))
    study.set_user_attr("space__fine_tune_batch_size", str(bounds.batch_size_choices))
    study.set_user_attr(
        "space__lora",
        f"r={bounds.lora_r_choices} alpha={bounds.lora_alpha_choices} "
        f"dropout[0,{bounds.lora_dropout_max}] targets={bounds.lora_target_modules_choices}",
    )
    study.set_user_attr(
        "space__optimizer",
        f"weight_decay[0,{bounds.weight_decay_max}] warmup[0,{bounds.warmup_ratio_max}] "
        f"scheduler={bounds.lr_scheduler_choices} grad_accum={bounds.grad_accum_choices}",
    )
    study.set_user_attr("space__cross_learning", str(bounds.cross_learning_choices))
    study.set_user_attr("space__target_scaler", str(bounds.target_scaler_choices))
    study.set_user_attr(
        "space__eval_metric",
        (
            f"fixed={bounds.eval_metric} (not searched; inert for mard_avg)"
            if bounds.eval_metric is not None
            else str(bounds.eval_metric_choices)
        ),
    )
    study.set_user_attr("space__eval_steps", str(bounds.eval_steps))
    # Reflect the EFFECTIVE covariate pools: the config can restrict
    # ``past_candidates`` and, when ``insulin_as_known`` is set, insulin channels
    # move out of the past pool into their own known-future ``use_known__*``
    # toggles (so they never appear as ``use_past__*`` parameters).
    insulin_candidates = [
        col
        for col in bounds.past_candidates
        if PAST_COVARIATE_MASKS.get(col) == INSULIN_OBSERVED_COL
    ]
    known_desc = "hour_sin, hour_cos (on/off together)"
    if bounds.insulin_as_known:
        past_pool = [
            col for col in bounds.past_candidates if col not in set(insulin_candidates)
        ]
        if insulin_candidates:
            known_desc += (
                "; insulin as known-future (per-channel use_known__*): "
                f"{insulin_candidates}"
            )
    else:
        past_pool = list(bounds.past_candidates)
    study.set_user_attr("space__past_covariates", str(past_pool))
    study.set_user_attr(
        "space__static_covariates",
        str([*STATIC_COVARIATE_CANDIDATES, *PIECEWISE_STATIC_CANDIDATES]),
    )
    study.set_user_attr("space__known_covariates", known_desc)
    study.set_user_attr("space__insulin_as_known", str(bounds.insulin_as_known))

    # Optional richer Markdown note when optuna-dashboard is installed.
    try:  # pragma: no cover - optional dependency
        import optuna_dashboard

        optuna_dashboard.save_note(
            study,
            "## Chronos-2 x Optuna (MetaboNet)\n"
            f"**Objective:** {objective_desc} on the internal temporal "
            "validation split.\n\n"
            "**Search:** LoRA fine-tuning of Chronos-2 over context length, LoRA "
            "config, optimizer/scheduler, covariates, and target scaling.\n\n"
            "Per-horizon RMSE/MARD/DTS (30/60/90/120) are on each trial's "
            "*User Attributes* tab.",
        )
    except Exception:
        pass


def build_objective(
    config: Dict[str, Any],
    data: Any,
    bounds: SearchSpaceBounds,
    gpu_label: str,
    metric_names: Sequence[str],
    is_multi: bool,
) -> Chronos2Objective:
    return Chronos2Objective(
        data,
        config["output_root"],
        bounds=bounds,
        time_limit=config.get("time_limit"),
        keep_models=bool(config.get("keep_models", False)),
        gpu_label=gpu_label,
        primary_metric=metric_names[0],
        objective_metrics=list(metric_names) if is_multi else None,
        enable_pruning=bool(config.get("enable_pruning", False)),
        report_intermediate=bool(config.get("report_intermediate", True)),
        early_stopping_patience=config.get("fine_tune_early_stopping_patience"),
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
