# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""gbm x Optuna sweep worker.

Supplies the gbm search space, objective, and dashboard metadata to the shared
sweep engine (:mod:`src.workflows.sweeps.engine`).

Usage:
    python -m src.workflows.sweeps.optuna_gbm.study create --config <yaml>
    python -m src.workflows.sweeps.optuna_gbm.study run --config <yaml> \\
        --n-trials 200 --gpu-label 0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import optuna

from ..engine import StudySpec, _parse_objectives, run_main
from ..optuna_chronos2.data import build_data
from .objective import GBMObjective
from .search_space import GBMSearchSpaceBounds

logger = logging.getLogger(__name__)

DEFAULT_CONFIG: Dict[str, Any] = {
    "study_name": "gbm_metabonet_poc",
    "storage": "sqlite:///results/optuna/gbm_metabonet_poc.db",
    "processed_path": "cache/data/metabonet/processed",
    "output_root": "results/optuna/gbm_metabonet_poc_models",
    "primary_metric": "mard_avg",
    "direction": "minimize",
    "time_limit": None,
    "timeout": None,
    "keep_models": False,
    "num_cpus": None,  # per-worker CPU budget (None -> AutoGluon auto-detect)
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
        "num_cpus": 2,
        "data": {
            "patient_limit": 8,
            "max_val_windows_per_patient": 3,
            "max_val_windows_total": 18,
        },
        "search_space": {
            "num_boost_round": 200,
        },
    },
}


def build_bounds(config: Dict[str, Any]) -> GBMSearchSpaceBounds:
    return GBMSearchSpaceBounds(**config.get("search_space", {}))


def _apply_dashboard_metadata(study: optuna.Study, config: Dict[str, Any]) -> None:
    """Attach a compact human-readable summary of the GBM search space."""
    metric_names, directions = _parse_objectives(config)
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001 - older storages may not support it
        pass

    bounds = build_bounds(config)
    data = config.get("data", {})
    objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
    study.set_user_attr("model", "GBM (RecursiveTabular/DirectTabular)")
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
    study.set_user_attr("space__strategy", str(bounds.strategy_choices))
    study.set_user_attr("space__backend", str(bounds.backend_choices))
    study.set_user_attr("space__lag_set", str(bounds.lag_set_choices))
    study.set_user_attr("space__target_scaler", str(bounds.target_scaler_choices))
    study.set_user_attr(
        "space__use_first_difference", str(bounds.use_first_difference_choices)
    )
    study.set_user_attr(
        "space__num_boost_round",
        f"fixed={bounds.num_boost_round} (early stopping picks effective count)",
    )
    study.set_user_attr(
        "space__learning_rate",
        f"loguniform[{bounds.learning_rate_min}, {bounds.learning_rate_max}]",
    )
    study.set_user_attr(
        "space__lightgbm",
        f"num_leaves[{bounds.num_leaves_min}, {bounds.num_leaves_max}] "
        f"feature_fraction[{bounds.feature_fraction_min}, {bounds.feature_fraction_max}] "
        f"min_data_in_leaf[{bounds.min_data_in_leaf_min}, {bounds.min_data_in_leaf_max}] "
        f"bagging_fraction[{bounds.bagging_fraction_min}, {bounds.bagging_fraction_max}] "
        f"lambda_l1/l2 (log)",
    )
    study.set_user_attr(
        "space__catboost",
        f"depth[{bounds.depth_min}, {bounds.depth_max}] "
        f"l2_leaf_reg[{bounds.l2_leaf_reg_min}, {bounds.l2_leaf_reg_max}] "
        f"random_strength (log) bagging_temperature"
        f"[{bounds.bagging_temperature_min}, {bounds.bagging_temperature_max}]",
    )
    study.set_user_attr(
        "space__xgboost",
        f"max_depth[{bounds.max_depth_min}, {bounds.max_depth_max}] "
        f"min_child_weight[{bounds.min_child_weight_min}, {bounds.min_child_weight_max}] "
        f"subsample[{bounds.subsample_min}, {bounds.subsample_max}] "
        f"colsample_bytree[{bounds.colsample_bytree_min}, {bounds.colsample_bytree_max}] "
        f"reg_alpha/reg_lambda/gamma (log)",
    )
    study.set_user_attr(
        "space__eval_metric", f"fixed={bounds.eval_metric} (not searched)"
    )
    study.set_user_attr(
        "space__known_covariates",
        "hour_sin, hour_cos (on/off together); insulin as known-future "
        f"(per-channel use_known__*): {bounds.known_candidates}; "
        f"include_meal={bounds.include_meal}",
    )
    study.set_user_attr("space__static_covariates", str(bounds.static_candidates))


def build_objective(
    config: Dict[str, Any],
    data: Any,
    bounds: Any,
    gpu_label: str,
    metric_names: Sequence[str],
    is_multi: bool,
) -> Any:
    return GBMObjective(
        data,
        config["output_root"],
        bounds=bounds,
        time_limit=config.get("time_limit"),
        keep_models=bool(config.get("keep_models", False)),
        gpu_label=gpu_label,
        num_cpus=config.get("num_cpus"),
        primary_metric=metric_names[0],
        objective_metrics=list(metric_names) if is_multi else None,
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
