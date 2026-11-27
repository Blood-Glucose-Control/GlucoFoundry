# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""
Data cleaning utilities for the CGMacros dataset (PhysioNet, v1.0.0).

Each participant has one CSV with a row per minute. Most files have:
    Timestamp, Libre GL, Dexcom GL, HR, Calories (Activity), METs,
    Meal Type, Calories, Carbs, Protein, Fat, Fiber, Amount Consumed, Image path
but the columns vary between participants (checked on the v1.0.0 files):
- 11 files have ``Intensity`` (0-3) instead of ``METs``
- 1 file has ``Steps`` instead of ``Calories (Activity)``
- some have an extra ``Unnamed: 0`` index, ``Sugar`` or ``RecordIndex`` column
- ``Amount Consumed`` sometimes has a trailing space, and is missing in 2 files
Meal columns are only filled on the row where a meal was logged.

Pipeline Steps (per participant):
1. Strip column names and parse timestamps; keep only real glucose readings
   (see Notes)
2. Floor timestamps to a 5-min grid (same as Brown 2019)
3. Aggregate each bin (glucose = the real reading in the bin, HR/METs = mean,
   activity calories/steps = sum, intensity = max, macros = sum,
   meal metadata = first, photo paths = all, joined with ";")
4. Reindex onto a regular 5-min grid, from the first to the last glucose
   reading of either sensor
5. Rename to the standard schema (bg_mM, carbohydrate_g, msg_type, hr_bpm, ...).
   Both sensors are kept (bg_mgdL_dexcom, bg_mgdL_libre); bg_mM is Dexcom.
6. Run preprocessing_pipeline (adds COB, time-of-day; IOB is NaN since there is
   no insulin data)

Notes:
- No participant has type 1 diabetes: 15 normal, 16 prediabetes, 14 type 2.
  Nobody uses insulin, so there is no dose/basal data.
- Dexcom G6 reads every 5 min (abdomen); Libre Pro every 15 min (upper arm).
  The CSVs have both linearly interpolated to 1 minute. We recover the real
  readings (where the interpolated line bends) and drop the rest, so each
  5-min row holds the real reading taken in that bin, or NaN. Libre has a
  value in about 1 of every 3 rows.
- Meal types are normalized to breakfast / lunch / dinner / snack.
- Most meals have a second photo a few minutes later on a row with no meal
  label (usually the leftovers, sometimes a drink or food that was
  never logged). These stay on the row where they were taken, in
  ``unlabeled_image_paths``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ....utils.unit import mg_dl_to_mmol_l
from ...models import ColumnNames
from ...preprocessing.pipeline import preprocessing_pipeline
from ...utils.patient_id import format_patient_id

logger = logging.getLogger(__name__)

DATASET_NAME = "cgmacros"
INTERVAL = "5min"

# Folder inside cache/data/cgmacros/raw/ that holds CGMacros-XXX/ and bio.csv
RAW_SUBDIR = "CGMacros"

# Raw glucose column -> output column (mg/dL). bg_mM is computed from Dexcom.
SENSOR_COLUMNS: dict[str, str] = {
    "Dexcom GL": "bg_mgdL_dexcom",
    "Libre GL": "bg_mgdL_libre",
}
PRIMARY_SENSOR = "Dexcom GL"
# Minutes between real readings for each sensor
SENSOR_INTERVAL_MIN: dict[str, int] = {"Dexcom GL": 5, "Libre GL": 15}
# A real reading is where the interpolated line bends. With whole-number
# readings, a bend changes the slope by at least 1/15 mg/dL per minute, far
# above float noise.
BEND_TOL = 0.03

# Raw column names (after stripping whitespace)
RAW_COLS = {
    "timestamp": "Timestamp",
    "hr": "HR",
    "activity_calories": "Calories (Activity)",
    "steps": "Steps",
    "mets": "METs",
    "intensity": "Intensity",
    "meal_type": "Meal Type",
    "meal_calories": "Calories",
    "carbs": "Carbs",
    "protein": "Protein",
    "fat": "Fat",
    "fiber": "Fiber",
    "amount_consumed": "Amount Consumed",
    "image_path": "Image path",
}

# Dataset-specific output columns (not in ColumnNames)
METS_COL = "mets"
INTENSITY_COL = "activity_intensity"  # Fitbit intensity level 0-3
MEAL_TYPE_COL = "meal_type"
MEAL_CALORIES_COL = "meal_calories"
PROTEIN_COL = "protein_g"
FAT_COL = "fat_g"
FIBER_COL = "fiber_g"
AMOUNT_CONSUMED_COL = "amount_consumed"
IMAGE_PATH_COL = "image_path"
UNLABELED_IMAGE_PATHS_COL = "unlabeled_image_paths"

# Raw meal labels seen in v1.0.0: Breakfast/breakfast, Lunch/lunch, Dinner/dinner,
# Snack/snack/Snacks/snack 1
MEAL_TYPES = ("breakfast", "lunch", "dinner", "snack")


def normalize_meal_type(label: object) -> object:
    """Map raw meal labels to breakfast / lunch / dinner / snack (NaN stays NaN)."""
    if not isinstance(label, str):
        return label
    cleaned = label.strip().lower()
    for meal in MEAL_TYPES:
        if cleaned.startswith(meal):
            return meal
    return cleaned


# ==================== Raw loading ====================


def read_participant_csv(path: Path) -> pd.DataFrame:
    """Read one participant CSV with stripped column names and parsed timestamps."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]  # e.g. 'Amount Consumed '
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    if RAW_COLS["timestamp"] not in df.columns:
        raise ValueError(f"{path.name}: missing '{RAW_COLS['timestamp']}' column")
    df[RAW_COLS["timestamp"]] = pd.to_datetime(df[RAW_COLS["timestamp"]])

    # A few photo paths name files that aren't in the download; drop them
    col = RAW_COLS["image_path"]
    if col in df.columns:
        listed = df[col].dropna()
        missing = listed[~listed.map(lambda p: (path.parent / p).exists())]
        if not missing.empty:
            logger.warning(f"{path.name}: photos not found: {list(missing)}")
            df.loc[missing.index, col] = np.nan
    return df


def load_raw_cgmacros_data(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every participant CSV (``data_dir/CGMacros-XXX/CGMacros-XXX.csv``).

    Returns:
        Dict mapping standardized patient ID (e.g. ``cgm_1``) -> raw DataFrame.

    Raises:
        FileNotFoundError: If no participant CSVs are found.
    """
    files = sorted(Path(data_dir).glob("CGMacros-*/CGMacros-*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CGMacros-XXX/CGMacros-XXX.csv files found in {data_dir}. "
            "Refer to README for download instructions."
        )
    raw = {
        format_patient_id(DATASET_NAME, path.stem): read_participant_csv(path)
        for path in files
    }
    logger.info(f"CGMacros loaded raw data for {len(raw)} participants")
    return raw


# ==================== Cleaning ====================


def recover_real_readings(
    times: pd.Series, values: pd.Series, interval_min: int
) -> pd.Series:
    """Recover the sensor's real readings from a 1-min linearly interpolated series.

    Interpolated lines bend only at real readings, so each bend (and the ends of
    each unbroken stretch) is a reading. When three readings are collinear the
    middle one makes no bend; it is put back at ``interval_min`` steps, which is
    exact because the line passes through it.

    Returns:
        Series of real readings indexed by their (whole-minute) timestamps.
    """
    s = pd.Series(np.asarray(values, dtype=float), index=pd.DatetimeIndex(times))
    s = s.dropna()
    s = s[~s.index.duplicated()].sort_index()
    if s.empty:
        return s

    # Unbroken stretches of consecutive minutes
    minutes = ((s.index - s.index[0]) // pd.Timedelta(minutes=1)).to_numpy()
    stretch = np.cumsum(np.r_[True, np.diff(minutes) != 1])

    keep: list[int] = []
    start = 0
    for length in np.bincount(stretch)[1:]:
        v = s.to_numpy()[start : start + length]
        bends = np.flatnonzero(np.abs(np.diff(v, 2)) > BEND_TOL) + 1
        knots = np.unique(np.r_[0, bends, length - 1])
        for a, b in zip(knots[:-1], knots[1:]):
            keep.append(start + a)
            steps = round((b - a) / interval_min)
            keep.extend(start + a + round(j * (b - a) / steps) for j in range(1, steps))
        keep.append(start + knots[-1])
        start += length
    return s.iloc[np.unique(keep)]


def clean_participant_data(
    raw_df: pd.DataFrame, patient_id: str
) -> pd.DataFrame | None:
    """Convert one participant's per-minute CSV into the standard 5-min schema.

    Returns:
        DataFrame indexed by ``datetime`` with standard columns, or None if
        neither sensor has readings for this participant.
    """
    glucose_cols = [c for c in SENSOR_COLUMNS if c in raw_df.columns]
    if not glucose_cols or raw_df[glucose_cols].notna().sum().sum() == 0:
        logger.warning(f"{patient_id}: no glucose readings, skipping")
        return None

    df = raw_df.copy()
    dt_col = ColumnNames.DATETIME.value
    df[dt_col] = df[RAW_COLS["timestamp"]].dt.floor(INTERVAL)

    # Numeric columns: coerce, since meal fields can be blank strings
    numeric = [
        *glucose_cols,
        RAW_COLS["hr"],
        RAW_COLS["activity_calories"],
        RAW_COLS["steps"],
        RAW_COLS["mets"],
        RAW_COLS["intensity"],
        RAW_COLS["meal_calories"],
        RAW_COLS["carbs"],
        RAW_COLS["protein"],
        RAW_COLS["fat"],
        RAW_COLS["fiber"],
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Glucose: keep only the real readings, drop the interpolated minutes
    for col in glucose_cols:
        real = recover_real_readings(
            df[RAW_COLS["timestamp"]], df[col], SENSOR_INTERVAL_MIN[col]
        )
        df[col] = df[RAW_COLS["timestamp"]].map(real)

    is_meal = (
        df[RAW_COLS["meal_type"]].notna()
        if RAW_COLS["meal_type"] in df.columns
        else pd.Series(False, index=df.index)
    )
    df["_is_meal"] = is_meal
    # Photos: labeled ones go with their meal, unlabeled ones stay where taken
    has_photos = RAW_COLS["image_path"] in df.columns
    if has_photos:
        photo = df[RAW_COLS["image_path"]]
        df["_unlabeled_image_paths"] = photo.where(~is_meal)
        df[RAW_COLS["image_path"]] = photo.where(is_meal)
    if RAW_COLS["meal_type"] in df.columns:
        df[RAW_COLS["meal_type"]] = df[RAW_COLS["meal_type"]].map(normalize_meal_type)

    # Macros only count on meal rows
    macro_cols = [
        RAW_COLS[k] for k in ("meal_calories", "carbs", "protein", "fat", "fiber")
    ]
    for col in macro_cols:
        if col in df.columns:
            df[col] = df[col].where(is_meal)

    # Glucose: the real reading in the bin (averaged in the rare case of two)
    agg: dict[str, str] = {col: "mean" for col in glucose_cols}
    agg["_is_meal"] = "any"
    for key, how in [
        ("hr", "mean"),
        ("activity_calories", "sum"),
        ("steps", "sum"),
        ("mets", "mean"),
        ("intensity", "max"),
        ("meal_type", "first"),
        ("amount_consumed", "first"),
    ]:
        if RAW_COLS[key] in df.columns:
            agg[RAW_COLS[key]] = how
    for col in macro_cols:
        if col in df.columns:
            agg[col] = "sum"

    # min_count=1 keeps all-NaN bins as NaN instead of 0 for summed Fitbit columns
    binned = df.groupby(dt_col).agg(agg)
    for key in ("activity_calories", "steps"):
        if RAW_COLS[key] in df.columns:
            binned[RAW_COLS[key]] = df.groupby(dt_col)[RAW_COLS[key]].sum(min_count=1)
    # Photos: every path in the bin, joined with ";" (two meals can share a bin)
    if has_photos:
        for col in (RAW_COLS["image_path"], "_unlabeled_image_paths"):
            photos = df.dropna(subset=col)
            binned[col] = photos.groupby(dt_col)[col].agg(";".join)

    # Regular grid between the first and last reading of either sensor
    valid = binned.index[binned[glucose_cols].notna().any(axis=1)]
    grid = pd.date_range(valid.min(), valid.max(), freq=INTERVAL, name=dt_col)
    binned = binned.reindex(grid)
    is_meal_bin = binned["_is_meal"].fillna(False).astype(bool)

    out = pd.DataFrame(index=grid)
    out[ColumnNames.P_NUM.value] = patient_id
    for raw_col, out_col in SENSOR_COLUMNS.items():
        out[out_col] = binned[raw_col] if raw_col in binned.columns else np.nan
    out[ColumnNames.BG.value] = mg_dl_to_mmol_l(out, SENSOR_COLUMNS[PRIMARY_SENSOR])

    # Meals: carbs = 0 outside meal bins so COB sees "no food" rather than "unknown"
    carbs = binned.get(RAW_COLS["carbs"], pd.Series(np.nan, index=grid))
    out[ColumnNames.CARBOHYDRATE_G.value] = carbs.where(is_meal_bin, 0.0)
    out[ColumnNames.MSG_TYPE.value] = np.where(is_meal_bin, "ANNOUNCE_MEAL", "")

    # Fitbit (each participant has only some of these)
    renames = {
        "hr": ColumnNames.HR_BPM.value,
        "activity_calories": ColumnNames.CALS.value,
        "steps": ColumnNames.STEPS.value,
        "mets": METS_COL,
        "intensity": INTENSITY_COL,
    }
    for key, new in renames.items():
        if RAW_COLS[key] in binned.columns:
            out[new] = binned[RAW_COLS[key]]

    # Extra meal detail (meal bins only)
    extras = {
        "meal_type": MEAL_TYPE_COL,
        "meal_calories": MEAL_CALORIES_COL,
        "protein": PROTEIN_COL,
        "fat": FAT_COL,
        "fiber": FIBER_COL,
        "amount_consumed": AMOUNT_CONSUMED_COL,
        "image_path": IMAGE_PATH_COL,
    }
    for key, new in extras.items():
        if RAW_COLS[key] in binned.columns:
            out[new] = binned[RAW_COLS[key]].where(is_meal_bin)
    if has_photos:
        out[UNLABELED_IMAGE_PATHS_COL] = binned["_unlabeled_image_paths"]

    return out


def process_single_patient_data(args: tuple) -> tuple[str, pd.DataFrame | None]:
    """Clean and preprocess one participant (top-level so it can run in a pool).

    Args:
        args: Tuple of (patient_id, raw_df).

    Returns:
        (patient_id, processed_df), where processed_df is None if the
        participant has no glucose readings.
    """
    patient_id, raw_df = args
    cleaned = clean_participant_data(raw_df, patient_id)
    if cleaned is None:
        return patient_id, None
    try:
        processed = preprocessing_pipeline(
            patient_id, cleaned, use_aggregation=False, basal_delivery_type=None
        )
    except Exception as e:
        logger.warning(
            f"Patient {patient_id} preprocessing failed: {e}. Using cleaned data."
        )
        processed = cleaned
    return patient_id, processed
