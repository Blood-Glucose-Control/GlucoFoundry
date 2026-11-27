# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Model-agnostic per-trial objective for the nocturnal HPO sweeps.

Each trial: build a model from the family's suggested hyperparameters, fit it on
the combined train split (skipped for zero-shot models), evaluate it on the
midnight-anchored holdout episodes via
:func:`experiments.nocturnal.eval.evaluate_nocturnal_forecasting`, and return the
primary reported metric (WQL by default). RMSE / Brier@3.9 / MACE / coverage /
DILATE are logged as trial user attributes.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Sequence

import optuna

from experiments.nocturnal.eval import evaluate_nocturnal_forecasting
from experiments.nocturnal.hpo._shared.data import NocturnalData
from src.workflows.forecasting.modeling import create_model_and_config

logger = logging.getLogger(__name__)

# Maps an engine metric name (e.g. "wql") to the key returned by
# evaluate_nocturnal_forecasting (e.g. "overall_wql").
_METRIC_KEYS: Dict[str, str] = {
    "wql": "overall_wql",
    "rmse": "overall_rmse",
    "brier": "overall_brier",
    "mace": "overall_mace",
}

# Non-scalar result keys (arrays / per-patient records) that must not be logged
# as Optuna user attrs.
_NON_SCALAR_RESULT_KEYS = frozenset({"per_patient", "per_episode", "quantile_levels"})


class NocturnalObjective:
    """Callable Optuna objective that trains + scores one config per trial."""

    def __init__(
        self,
        data: NocturnalData,
        bounds: Any,
        output_root: str | Path,
        *,
        model_type: str,
        forecast_length: int,
        primary_metric: str = "wql",
        gpu_label: str = "",
        compute_dilate: bool = False,
        keep_models: bool = False,
    ) -> None:
        self.data = data
        self.bounds = bounds
        self.output_root = Path(output_root)
        self.model_type = model_type
        self.forecast_length = int(forecast_length)
        self.primary_metric = primary_metric
        self.gpu_label = gpu_label
        self.compute_dilate = bool(compute_dilate)
        self.keep_models = bool(keep_models)

    def _result_key(self, metric: str) -> str:
        return _METRIC_KEYS.get(metric, f"overall_{metric}")

    def __call__(self, trial: optuna.Trial) -> float:
        point: Dict[str, Any] = self.bounds.suggest(trial)
        context_length = int(point["context_length"])
        covariate_cols: Sequence[str] = point.get("covariate_cols") or []

        model_kwargs = dict(point)
        model_kwargs.pop("covariate_cols", None)
        model_kwargs["context_length"] = context_length
        model_kwargs["forecast_length"] = self.forecast_length
        if covariate_cols:
            model_kwargs["covariate_cols"] = list(covariate_cols)

        model, _config = create_model_and_config(self.model_type, **model_kwargs)
        trial.set_user_attr("context_length", context_length)
        trial.set_user_attr("covariate_cols", list(covariate_cols))
        trial.set_user_attr("gpu", self.gpu_label)

        trial_dir = self.output_root / f"trial_{trial.number:05d}"
        try:
            if not getattr(model, "supports_zero_shot", False):
                model.fit(train_data=self.data.train, output_dir=str(trial_dir))

            results = evaluate_nocturnal_forecasting(
                model=model,
                holdout_data=self.data.holdout,
                context_length=context_length,
                forecast_length=self.forecast_length,
                covariate_cols=list(covariate_cols) or None,
                probabilistic=True,
                compute_dilate=self.compute_dilate,
            )
        finally:
            if not self.keep_models and trial_dir.exists():
                shutil.rmtree(trial_dir, ignore_errors=True)

        # Log every scalar summary metric the evaluator produced (overall_* plus
        # coverage/sharpness/DILATE/episode counts) so stored trial metadata
        # matches evaluate_nocturnal_forecasting's output. Non-scalar records
        # (per-patient/per-episode frames, quantile_levels) are skipped.
        for key, value in results.items():
            if key in _NON_SCALAR_RESULT_KEYS:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                trial.set_user_attr(key, float(value))

        primary_key = self._result_key(self.primary_metric)
        if primary_key not in results or results[primary_key] is None:
            raise optuna.TrialPruned(
                f"Model produced no '{primary_key}' (needs probabilistic forecasts "
                f"for {self.primary_metric})."
            )
        return float(results[primary_key])


def build_objective(
    config: Dict[str, Any],
    data: NocturnalData,
    bounds: Any,
    gpu_label: str,
    metric_names: Sequence[str],
    is_multi: bool,
    *,
    model_type: str,
) -> NocturnalObjective:
    """Construct a :class:`NocturnalObjective` from engine-provided arguments."""
    data_cfg = config.get("data", {}) or {}
    return NocturnalObjective(
        data,
        bounds,
        config["output_root"],
        model_type=model_type,
        forecast_length=int(data_cfg.get("forecast_length", 96)),
        primary_metric=metric_names[0],
        gpu_label=gpu_label,
        compute_dilate=bool(config.get("compute_dilate", False)),
        keep_models=bool(config.get("keep_models", False)),
    )
