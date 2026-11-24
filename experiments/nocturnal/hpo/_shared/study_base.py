# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Boilerplate helpers shared by every nocturnal family's ``study.py``.

Each family supplies only its ``MODEL_TYPE``, a search-space bounds dataclass,
and any family-specific config overrides; :func:`make_study` wires those into a
:class:`~src.workflows.hpo.engine.StudySpec` with the shared data builder,
objective, and dashboard metadata.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Sequence, Type

import optuna

from experiments.nocturnal.hpo._shared.data import build_data as _build_data
from experiments.nocturnal.hpo._shared.objective import (
    build_objective as _build_objective,
)
from src.experiments.registry import get_experiment_paths
from src.workflows.hpo.engine import StudySpec, _parse_objectives

EXPERIMENT = "nocturnal"


def _experiment_storage() -> str:
    """Shared nocturnal journal DB under the experiment's own results/ dir.

    Per RFC-001, run artifacts live inside ``experiments/<name>/`` (redirectable
    to scratch via ``GLUCOFOUNDRY_ARTIFACTS_ROOT``), not the shared repo-root
    ``results/`` — so experiments stay self-contained and never collide.
    """
    results = get_experiment_paths(EXPERIMENT).results
    return str(results / "optuna" / "nocturnal_hpo.journal")


_BASE_CONFIG: Dict[str, Any] = {
    "storage_backend": "journal",
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
        "data": {"datasets": ["aleppo_2017"], "patient_limit": 2},
        "search_space": {},
    },
}


def base_default_config(
    study_name: str,
    *,
    output_root: Optional[str] = None,
    search_space: Optional[Dict[str, Any]] = None,
    smoke_search_space: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a family DEFAULT_CONFIG from the shared base + family specifics.

    Storage and per-trial model output both resolve under the nocturnal
    experiment's own artifact dirs (``experiments/nocturnal/{results,trained_models}/``).
    """
    config = copy.deepcopy(_BASE_CONFIG)
    config["study_name"] = study_name
    config["storage"] = _experiment_storage()
    if output_root is None:
        trained_models = get_experiment_paths(EXPERIMENT).trained_models
        output_root = str(trained_models / f"nocturnal_{study_name}_models")
    config["output_root"] = output_root
    if search_space:
        config["search_space"] = search_space
    if smoke_search_space is not None:
        config["smoke"]["search_space"] = smoke_search_space
    if overrides:
        config.update(overrides)
    return config


def _dashboard_fn(model_name: str, bounds_cls: Type[Any]):
    def _apply(study: optuna.Study, config: Dict[str, Any]) -> None:
        metric_names, directions = _parse_objectives(config)
        try:
            study.set_metric_names(list(metric_names))
        except Exception:  # noqa: BLE001 - older storages may not support it
            pass
        bounds = bounds_cls(**config.get("search_space", {}))
        data = config.get("data", {})
        objective_desc = ", ".join(f"{d} {m}" for m, d in zip(metric_names, directions))
        study.set_user_attr("model", model_name)
        study.set_user_attr(
            "objective", f"{objective_desc} on midnight-anchored nocturnal holdout"
        )
        study.set_user_attr("datasets", ", ".join(data.get("datasets", [])))
        study.set_user_attr(
            "forecast",
            f"{data.get('forecast_length', 96)}-step overnight forecast (+6h)",
        )
        study.set_user_attr("search_space", repr(vars(bounds)))
        study.set_user_attr(
            "logged_metrics", "rmse/wql/brier/mace stored as trial user attrs"
        )

    return _apply


def make_study(
    model_type: str,
    default_config: Dict[str, Any],
    bounds_cls: Type[Any],
    model_name: str,
) -> StudySpec:
    """Wire a family's config + search space into a shared-objective StudySpec."""

    def build_bounds(config: Dict[str, Any]) -> Any:
        return bounds_cls(**config.get("search_space", {}))

    def build_objective(
        config: Dict[str, Any],
        data: Any,
        bounds: Any,
        gpu_label: str,
        metric_names: Sequence[str],
        is_multi: bool,
    ) -> Any:
        return _build_objective(
            config,
            data,
            bounds,
            gpu_label,
            metric_names,
            is_multi,
            model_type=model_type,
        )

    return StudySpec(
        default_config=default_config,
        build_bounds=build_bounds,
        build_objective=build_objective,
        build_data=_build_data,
        apply_dashboard_metadata=_dashboard_fn(model_name, bounds_cls),
    )
