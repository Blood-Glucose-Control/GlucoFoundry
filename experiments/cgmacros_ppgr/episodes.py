# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Meal-anchored episodes for the CGMacros PPGR task (GlucoFM paper, App. D.2).

For each logged meal and each sensor:

* input: the 24 h before the exact meal minute, as 288 five-minute slots
  counted back from the meal (slot 287 ends at the meal). Each real reading
  goes in its slot (averaged if two); gaps between readings are filled
  linearly and the edges with the nearest value. Only readings at or before
  the meal are used.
* meal-start glucose: the latest reading at or before the meal, at most
  5 min (Dexcom) / 15 min (Libre) old.
* targets: glucose change from meal start at +5..+120 min (Dexcom, 24 values)
  or +15..+120 min (Libre, 8 values), using the nearest reading within
  2.5 / 7.5 min of each horizon, without interpolation.

A meal is kept only if no other meal is logged in the next 120 min, both
sensors cover the 24 h before it with no gap over 60 min, and both sensors
have a meal-start glucose and every target ("same meals per sensor").

Episodes are built from the raw per-minute CGMacros CSVs rather than the
processed 5-min frames: glucose uses the real readings recovered from the
interpolated CSVs (``recover_real_readings``) and meal times keep their exact
minute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.diabetes_datasets.cgmacros.data_cleaner import (
    RAW_COLS,
    SENSOR_INTERVAL_MIN,
    recover_real_readings,
)

logger = logging.getLogger(__name__)

#: Sampling cadence of the input series, in minutes.
SAMPLING_INTERVAL_MINUTES = 5
CONTEXT_STEPS = 288  # 24 h at 5 min, the CGM-JEPA pretraining window
CONTEXT_MIN = CONTEXT_STEPS * SAMPLING_INTERVAL_MINUTES
RESPONSE_MIN = 120  # prediction window, and no other meal allowed inside it
MAX_GAP_MIN = 60  # longest gap allowed between readings in the 24 h input


@dataclass(frozen=True)
class SensorSpec:
    raw_col: str
    interval_min: int  # minutes between real readings = target spacing
    max_baseline_age_min: float
    target_tol_min: float

    @property
    def horizons_min(self) -> np.ndarray:
        return np.arange(self.interval_min, RESPONSE_MIN + 1, self.interval_min)


SENSORS: dict[str, SensorSpec] = {
    "dexcom": SensorSpec("Dexcom GL", SENSOR_INTERVAL_MIN["Dexcom GL"], 5, 2.5),
    "libre": SensorSpec("Libre GL", SENSOR_INTERVAL_MIN["Libre GL"], 15, 7.5),
}


@dataclass
class MealEvents:
    """All kept meals for one sensor (row i is the same meal in every sensor)."""

    patient_ids: np.ndarray  # (N,) str
    meal_times: np.ndarray  # (N,) datetime64[m]
    contexts: np.ndarray  # (N, 288) mg/dL
    baselines: np.ndarray  # (N,) mg/dL at meal start
    targets: np.ndarray  # (N, H) change from meal start, mg/dL
    horizons_min: np.ndarray  # (H,)


def _to_minutes(times) -> np.ndarray:
    """Timestamps -> integer minutes since the epoch."""
    return np.asarray(times, dtype="datetime64[m]").astype(np.int64)


def meal_times(raw_df: pd.DataFrame) -> np.ndarray:
    """Exact minutes (since epoch) of every logged meal, sorted and unique."""
    col = RAW_COLS["meal_type"]
    if col not in raw_df.columns:
        return np.empty(0, dtype=np.int64)
    rows = raw_df[col].notna()
    return np.unique(_to_minutes(raw_df.loc[rows, RAW_COLS["timestamp"]]))


def isolated_meals(meals: np.ndarray) -> np.ndarray:
    """Meals with no other meal in the next (0, RESPONSE_MIN] minutes."""
    next_gap = np.r_[np.diff(meals), np.inf]
    return meals[next_gap > RESPONSE_MIN]


def sensor_readings(
    raw_df: pd.DataFrame, spec: SensorSpec
) -> tuple[np.ndarray, np.ndarray]:
    """Real readings for one sensor as (minutes since epoch, mg/dL)."""
    if spec.raw_col not in raw_df.columns:
        return np.empty(0, dtype=np.int64), np.empty(0)
    real = recover_real_readings(
        raw_df[RAW_COLS["timestamp"]],
        pd.to_numeric(raw_df[spec.raw_col], errors="coerce"),
        spec.interval_min,
    )
    return _to_minutes(real.index.to_numpy()), real.to_numpy(dtype=float)


def build_context(t: np.ndarray, v: np.ndarray, meal: int) -> np.ndarray:
    """288-slot input ending at the meal minute (readings must be <= meal)."""
    age = meal - t  # minutes before the meal, 0 <= age < CONTEXT_MIN
    slot = CONTEXT_STEPS - 1 - age // SAMPLING_INTERVAL_MINUTES
    sums = np.bincount(slot, weights=v, minlength=CONTEXT_STEPS)
    counts = np.bincount(slot, minlength=CONTEXT_STEPS)
    ctx = pd.Series(np.where(counts > 0, sums / np.maximum(counts, 1), np.nan))
    ctx = ctx.interpolate(limit_area="inside").ffill().bfill()
    return ctx.to_numpy(dtype=np.float32)


def sensor_event(
    t: np.ndarray, v: np.ndarray, meal: int, spec: SensorSpec
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """(context, meal-start glucose, target changes) for one meal, or None."""
    in_window = (t > meal - CONTEXT_MIN) & (t <= meal)
    tw = t[in_window]
    if tw.size == 0:
        return None

    # 24 h coverage: no gap over MAX_GAP_MIN, including from the window start
    edges = np.r_[meal - CONTEXT_MIN, tw]
    if np.diff(edges).max() > MAX_GAP_MIN:
        return None

    # Meal-start glucose: latest reading at or before the meal, not too old
    if meal - tw[-1] > spec.max_baseline_age_min:
        return None
    baseline = float(v[in_window][-1])

    # Targets: nearest reading to each horizon within the tolerance
    targets = np.empty(len(spec.horizons_min))
    for i, h in enumerate(spec.horizons_min):
        dist = np.abs(t - (meal + h))
        j = int(np.argmin(dist))
        if dist[j] > spec.target_tol_min:
            return None
        targets[i] = v[j] - baseline

    return build_context(tw, v[in_window], meal), baseline, targets


def build_episodes(raw: dict[str, pd.DataFrame]) -> dict[str, MealEvents]:
    """Kept meals for every sensor, from raw per-participant CGMacros frames.

    Args:
        raw: Participant id -> raw per-minute CSV frame, as returned by
            ``load_raw_cgmacros_data``.

    Returns:
        Sensor name (``"dexcom"``, ``"libre"``) -> :class:`MealEvents`; row ``i``
        is the same meal for both sensors.
    """
    rows: dict[str, list] = {name: [] for name in SENSORS}
    for pid, raw_df in raw.items():
        readings = {
            name: sensor_readings(raw_df, spec) for name, spec in SENSORS.items()
        }
        for meal in isolated_meals(meal_times(raw_df)):
            per_sensor = {
                name: sensor_event(*readings[name], meal, spec)
                for name, spec in SENSORS.items()
            }
            if any(ev is None for ev in per_sensor.values()):
                continue
            for name, ev in per_sensor.items():
                assert ev is not None
                rows[name].append((pid, meal, *ev))

    events = {}
    for name, spec in SENSORS.items():
        r = rows[name]
        events[name] = MealEvents(
            patient_ids=np.array([x[0] for x in r], dtype=str),
            meal_times=np.array([x[1] for x in r], dtype="datetime64[m]"),
            contexts=np.stack([x[2] for x in r]) if r else np.empty((0, CONTEXT_STEPS)),
            baselines=np.array([x[3] for x in r], dtype=float),
            targets=np.stack([x[4] for x in r])
            if r
            else np.empty((0, len(spec.horizons_min))),
            horizons_min=spec.horizons_min,
        )
    n = len(events["dexcom"].patient_ids)
    logger.info(
        "PPGR: %d meals from %d participants",
        n,
        len(set(events["dexcom"].patient_ids)),
    )
    return events


def save_events(events: dict[str, MealEvents], path: Path) -> None:
    arrays = {
        f"{name}/{field}": getattr(ev, field)
        for name, ev in events.items()
        for field in MealEvents.__dataclass_fields__
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_events(path: Path) -> dict[str, MealEvents]:
    with np.load(path) as blob:
        return {
            name: MealEvents(
                **{f: blob[f"{name}/{f}"] for f in MealEvents.__dataclass_fields__}
            )
            for name in SENSORS
        }
