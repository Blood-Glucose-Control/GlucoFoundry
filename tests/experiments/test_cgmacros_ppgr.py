# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Tests for the CGMacros PPGR experiment: meal events, metrics, registration."""

import json

import numpy as np
import pandas as pd
import pytest

from experiments.cgmacros_ppgr.episodes import (
    CONTEXT_STEPS,
    SENSORS,
    build_context,
    build_episodes,
    sensor_event,
)
from experiments.cgmacros_ppgr.eval import positive_iauc, ppgr_mae
from experiments.cgmacros_ppgr.summarize import (
    EXPERIMENT_TYPE,
    CgmacrosPpgrSummarizer,
)
from src.experiments.registry import (
    available_experiments,
    discover_experiments,
    get_experiment,
    reset_registry,
)

DEX = SENSORS["dexcom"]
MEAL = 10_000  # minutes since epoch


def _dexcom(start: int, end: int, every: int = 5, offset: int = 2):
    """Readings every ``every`` min (off the clock grid), value = 100 + minute offset."""
    t = np.arange(start + offset, end, every)
    return t, 100.0 + (t - MEAL) * 0.1


def test_sensor_event_targets_and_baseline():
    t, v = _dexcom(MEAL - 1500, MEAL + 200)
    event = sensor_event(t, v, MEAL, DEX)
    assert event is not None
    ctx, baseline, targets = event

    assert baseline == pytest.approx(v[t <= MEAL][-1])  # reading at MEAL-3
    assert len(targets) == 24
    # Readings sit at MEAL+2, +7, +12...; horizon +5 -> nearest is +7 (2 min away)
    assert targets[0] == pytest.approx((100 + 0.7) - baseline)
    assert ctx.shape == (CONTEXT_STEPS,)


def test_context_uses_only_pre_meal_readings():
    t, v = _dexcom(MEAL - 1500, MEAL + 200)
    v = v.copy()
    v[t > MEAL] = 999.0  # would show up if post-meal readings leaked in
    event = sensor_event(t, v, MEAL, DEX)
    assert event is not None
    assert event[0].max() < 999.0


def test_build_context_slots_and_fill():
    # Readings 1 and 11 min before the meal -> last slot and two slots earlier
    ctx = build_context(np.array([MEAL - 11, MEAL - 1]), np.array([100.0, 120.0]), MEAL)
    assert ctx[-1] == 120.0
    assert ctx[-3] == 100.0
    assert ctx[-2] == pytest.approx(110.0)  # filled linearly
    assert ctx[0] == 100.0  # edge filled with the nearest value


@pytest.mark.parametrize(
    "case",
    ["stale_baseline", "missing_target", "long_gap"],
)
def test_sensor_event_rejects(case):
    t, v = _dexcom(MEAL - 1500, MEAL + 200)
    keep = np.ones(len(t), bool)
    if case == "stale_baseline":
        keep &= ~((t > MEAL - 10) & (t <= MEAL))  # latest pre-meal reading >5 min old
    elif case == "missing_target":
        keep &= ~((t > MEAL + 55) & (t < MEAL + 65))  # nothing near +60 min
    else:
        keep &= ~((t > MEAL - 600) & (t < MEAL - 520))  # 80-min hole in the input
    assert sensor_event(t[keep], v[keep], MEAL, DEX) is None


def test_build_episodes_drops_meal_followed_by_another():
    start = pd.Timestamp("2025-01-01")
    ts = pd.date_range(start, periods=3 * 24 * 60, freq="1min")
    minute = np.arange(len(ts))
    df = pd.DataFrame(
        {
            "Timestamp": ts,
            "Dexcom GL": 100 + 10 * np.sin(minute / 100),
            "Libre GL": 95 + 10 * np.sin(minute / 100),
            "Meal Type": None,
        }
    )
    # Keep only real-looking readings: Dexcom every 5 min, Libre every 15, interpolated
    for col, every in (("Dexcom GL", 5), ("Libre GL", 15)):
        df[col] = df[col].where(minute % every == 0).interpolate()
    day2 = 24 * 60 + 12 * 60
    df.loc[day2, "Meal Type"] = "Lunch"  # isolated
    df.loc[day2 + 300, "Meal Type"] = "Snack"  # followed by dinner 90 min later
    df.loc[day2 + 390, "Meal Type"] = "Dinner"  # isolated

    events = build_episodes({"cgm_1": df})
    kept = events["dexcom"].meal_times
    assert len(kept) == 2
    assert np.array_equal(kept, events["libre"].meal_times)  # same meals per sensor
    assert events["libre"].targets.shape == (2, 8)


def test_metrics_hand_example():
    h = np.array([15, 30])
    true = np.array([[30.0, 60.0]])
    pred = np.array([[60.0, 30.0]])
    # iAUC (true): 0.25 h * (0+30)/2 + 0.25 h * (30+60)/2 = 3.75 + 11.25 = 15
    assert positive_iauc(true, h)[0] == pytest.approx(15.0)
    out = ppgr_mae(pred, true, h)
    assert out["trajectory"] == pytest.approx(30.0)
    assert out["iauc_pos"] == pytest.approx(abs(18.75 - 15.0))  # pred: 7.5 + 11.25
    assert out["peak_rise"] == 0.0
    assert out["peak_time"] == 15.0


def test_recipe_is_discovered_and_registered():
    reset_registry()
    try:
        discover_experiments(force=True)
        assert "cgmacros_ppgr" in available_experiments()
        spec = get_experiment("cgmacros_ppgr")
        assert spec.summarizer is CgmacrosPpgrSummarizer
        # Not compatible with the forecasting workflow's hooks; see register.py
        assert spec.build_episodes is None
        assert spec.evaluate is None
    finally:
        reset_registry()


def test_summarizer_reads_run_results(tmp_path):
    def endpoint(value):
        return {
            m: {"mean": value, "sd": 0.1}
            for m in ("trajectory", "iauc_pos", "peak_rise", "peak_time")
        }

    run_dir = (
        tmp_path / EXPERIMENT_TYPE / "288ctx_24fh" / "cgm_jepa"
    ) / "2026-01-01_120000_cgmacros_frozen_all"
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(
        json.dumps(
            {
                "args": {"stage": "frozen", "participants": "all", "seeds": 10},
                "meals": 900,
                "summary": {
                    "mean": {
                        "model": endpoint(25.0),
                        "zero": endpoint(30.0),
                        "mean_curve": endpoint(28.0),
                    }
                },
            }
        )
    )
    df = CgmacrosPpgrSummarizer(tmp_path).summarize(metric="mae")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["model"] == "cgm_jepa" and row["stage"] == "frozen"
    assert row["mae"] == 25.0 and row["zero_mae"] == 30.0
