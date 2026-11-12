"""Tests for the CGMacros loader, using small synthetic CGMacros-shaped files."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cache_manager import CacheManager
from src.data.diabetes_datasets.cgmacros import cgmacros as cgmacros_module
from src.data.diabetes_datasets.cgmacros.cgmacros import CGMacrosDataLoader
from src.data.diabetes_datasets.cgmacros.data_cleaner import (
    clean_participant_data,
    read_participant_csv,
    recover_real_readings,
)

START = pd.Timestamp("2025-03-01 06:00:00")
MEAL_TIME = START + pd.Timedelta(minutes=62)  # falls in the 07:00 bin


def _participant_frame() -> pd.DataFrame:
    """Six hours of per-minute rows like a real CGMacros CSV, with one meal."""
    ts = pd.date_range(START, periods=6 * 60, freq="1min")
    minute = np.arange(len(ts))
    df = pd.DataFrame(
        {
            "Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "Libre GL": np.where(minute % 15 == 0, 95.0 + minute * 0.1, np.nan),
            "Dexcom GL": np.where(minute % 5 == 0, 100.0 + minute * 0.1, np.nan),
            "HR": 70.0,
            "Calories (Activity)": 1.5,
            "METs": 1.0,
            "Meal Type": None,
            "Calories": np.nan,
            "Carbs": np.nan,
            "Protein": np.nan,
            "Fat": np.nan,
            "Fiber": np.nan,
            "Amount Consumed ": np.nan,  # trailing space, as in the real files
            "Image path": None,
        }
    )
    row = int((MEAL_TIME - START) / pd.Timedelta(minutes=1))
    df.loc[row, ["Meal Type", "Image path"]] = ["Lunch", "photos/lunch_1.jpg"]
    df.loc[row, ["Calories", "Carbs", "Protein", "Fat", "Fiber"]] = [650, 80, 30, 20, 8]
    # Unlabeled photos: two in the 07:15 bin, one later in the 08:30 bin
    df.loc[row + 14, "Image path"] = "photos/drink.jpg"
    df.loc[row + 17, "Image path"] = "photos/leftovers.jpg"
    df.loc[row + 90, "Image path"] = "photos/later.jpg"
    return df


def _write_photos(raw: pd.DataFrame, folder: Path) -> None:
    """Create the photo files the frame refers to."""
    for p in raw["Image path"].dropna():
        (folder / p).parent.mkdir(parents=True, exist_ok=True)
        (folder / p).touch()


def _clean(raw: pd.DataFrame, tmp_path: Path) -> pd.DataFrame:
    path = tmp_path / "CGMacros-001.csv"
    raw.to_csv(path, index=False)
    _write_photos(raw, tmp_path)
    out = clean_participant_data(read_participant_csv(path), "cgm_1")
    assert out is not None
    return out


def test_clean_participant_data(tmp_path: Path):
    out = _clean(_participant_frame(), tmp_path)

    assert out.index.name == "datetime"
    assert pd.infer_freq(out.index) == "5min"
    # Both sensors kept; bg_mM comes from Dexcom
    assert out["bg_mgdL_dexcom"].iloc[0] == 100.0
    assert out["bg_mgdL_libre"].iloc[0] == 95.0
    assert out["bg_mM"].iloc[0] == pytest.approx(100.0 / 18.018, abs=0.01)

    meal_bin = MEAL_TIME.floor("5min")
    meal = out.loc[meal_bin]
    assert meal["carbohydrate_g"] == 80
    assert meal["msg_type"] == "ANNOUNCE_MEAL"
    assert meal["protein_g"] == 30
    assert meal["meal_type"] == "lunch"
    assert meal["image_path"] == "photos/lunch_1.jpg"
    assert (out.drop(index=meal_bin)["carbohydrate_g"] == 0).all()
    assert out["image_path"].notna().sum() == 1  # labeled photo on the meal row only

    # Unlabeled photos stay where they were taken
    unlabeled = out["unlabeled_image_paths"].dropna()
    assert unlabeled.to_dict() == {
        pd.Timestamp("2025-03-01 07:15"): "photos/drink.jpg;photos/leftovers.jpg",
        pd.Timestamp("2025-03-01 08:30"): "photos/later.jpg",
    }

    # Fitbit: HR averaged, activity calories summed over the 5 minutes
    assert out["hr_bpm"].iloc[0] == 70
    assert out["calories"].iloc[0] == pytest.approx(7.5)


def test_missing_photos_are_dropped(tmp_path: Path):
    raw = _participant_frame()
    raw.to_csv(tmp_path / "CGMacros-001.csv", index=False)
    _write_photos(raw, tmp_path)
    (tmp_path / "photos/drink.jpg").unlink()  # listed in the CSV, not on disk

    df = read_participant_csv(tmp_path / "CGMacros-001.csv")
    assert "photos/drink.jpg" not in set(df["Image path"].dropna())
    assert "photos/leftovers.jpg" in set(df["Image path"].dropna())


def test_recover_real_readings():
    """Readings off the clock grid, with 3 collinear ones, survive 1-min interpolation."""
    real = pd.Series(
        [100.0, 104.0, 108.0, 112.0, 109.0, 111.0],  # 100-104-108 have no bend
        index=START + pd.to_timedelta([2, 7, 12, 17, 22, 27], unit="min"),
    )
    minutes = pd.date_range(real.index[0], real.index[-1], freq="1min")
    interpolated = real.reindex(minutes).interpolate("time")

    out = recover_real_readings(minutes.to_series(), interpolated, interval_min=5)
    pd.testing.assert_series_equal(out, real, check_freq=False, check_names=False)


def test_clean_participant_data_column_variants(tmp_path: Path):
    """11 real files have Intensity instead of METs; 1 has Steps instead of calories."""
    raw = _participant_frame().rename(
        columns={"METs": "Intensity", "Calories (Activity)": "Steps"}
    )
    raw["Intensity"] = 2
    raw["Steps"] = 10
    out = _clean(raw, tmp_path)

    assert out["activity_intensity"].iloc[0] == 2
    assert out["step_count"].iloc[0] == 50  # summed over 5 minutes


def test_loader_processes_and_reloads_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cm = CacheManager(cache_root=str(tmp_path))
    root = cm.get_absolute_path_by_type("cgmacros", "raw") / "CGMacros"
    for num in (1, 2):
        folder = root / f"CGMacros-{num:03d}"
        folder.mkdir(parents=True)
        raw = _participant_frame()
        raw.to_csv(folder / f"CGMacros-{num:03d}.csv", index=False)
        _write_photos(raw, folder)
    monkeypatch.setattr(cgmacros_module, "get_cache_manager", lambda: cm)

    loader = CGMacrosDataLoader(use_cached=False, parallel=False)
    assert loader.processed_data is not None
    assert sorted(loader.processed_data) == ["cgm_1", "cgm_2"]
    df = loader.processed_data["cgm_1"]
    assert df["cob"].max() > 0  # COB computed from the logged meal
    assert df["iob"].isna().all()  # no insulin in CGMacros

    cached = CGMacrosDataLoader(use_cached=True)
    assert cached.processed_data is not None
    assert sorted(cached.processed_data) == ["cgm_1", "cgm_2"]
