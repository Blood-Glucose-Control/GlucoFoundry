# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Evaluation for the CGMacros PPGR task: endpoints and participant-level CV.

Endpoints (GlucoFM paper, App. D.2). All curves are glucose changes from meal
start (mg/dL) on a sensor's native horizons (Dexcom +5..+120 min, Libre
+15..+120 min):

* trajectory: MAE over all meals and horizons (mg/dL)
* positive iAUC: area of max(change, 0) over 0-2 h, trapezoid rule with a
  change of 0 prepended at t=0 (mg/dL*h)
* peak rise: largest change (mg/dL)
* peak time: minutes of the first largest change
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold

from .episodes import MealEvents
from .head import fit_predict_head

METRICS = ("trajectory", "iauc_pos", "peak_rise", "peak_time")
N_FOLDS = 5


def positive_iauc(delta: np.ndarray, horizons_min: np.ndarray) -> np.ndarray:
    """Per-meal positive incremental AUC in mg/dL*h. ``delta``: (N, H)."""
    y = np.maximum(np.c_[np.zeros(len(delta)), delta], 0.0)
    hours = np.r_[0.0, horizons_min] / 60.0
    return ((y[:, 1:] + y[:, :-1]) / 2 * np.diff(hours)).sum(axis=1)


def ppgr_mae(
    pred: np.ndarray, true: np.ndarray, horizons_min: np.ndarray
) -> dict[str, float]:
    """MAE of each endpoint between predicted and observed change curves."""
    h = np.asarray(horizons_min)
    return {
        "trajectory": float(np.abs(pred - true).mean()),
        "iauc_pos": float(
            np.abs(positive_iauc(pred, h) - positive_iauc(true, h)).mean()
        ),
        "peak_rise": float(np.abs(pred.max(axis=1) - true.max(axis=1)).mean()),
        "peak_time": float(
            np.abs(h[pred.argmax(axis=1)] - h[true.argmax(axis=1)]).mean()
        ),
    }


def evaluate(
    events: MealEvents, inputs: dict[str, np.ndarray], seeds: int
) -> dict[str, list[dict[str, float]]]:
    """Per-seed endpoint MAEs for each feature set's head and two baselines.

    Participants are split with ``GroupKFold(5)``, the same folds for every
    seed. Each seed trains one head per fold, and its held-out predictions from
    all folds are scored together.

    Args:
        events: Meals for one sensor.
        inputs: Name -> (N, F) features to train a head on (one per method).
        seeds: Number of head seeds.

    Returns:
        Method name -> one endpoint dict per seed. The baselines ``zero`` (no
        change) and ``mean_curve`` (training-fold mean response) do not depend
        on the seed, so they have a single entry.
    """
    folds = list(GroupKFold(N_FOLDS).split(events.targets, groups=events.patient_ids))
    y = events.targets
    results: dict[str, list[dict[str, float]]] = {name: [] for name in inputs}
    results.update({"mean_curve": [], "zero": []})

    zero = np.zeros_like(y)
    mean_curve = np.empty_like(y)
    for train, test in folds:
        mean_curve[test] = y[train].mean(axis=0)
    results["zero"].append(ppgr_mae(zero, y, events.horizons_min))
    results["mean_curve"].append(ppgr_mae(mean_curve, y, events.horizons_min))

    for name, x in inputs.items():
        for seed in range(seeds):
            pred = np.empty_like(y)
            for train, test in folds:
                pred[test] = fit_predict_head(
                    x[train], y[train], events.patient_ids[train], x[test], seed
                )
            results[name].append(ppgr_mae(pred, y, events.horizons_min))
    return results
