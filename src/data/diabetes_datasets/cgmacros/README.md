# CGMacros Dataset

## Overview

CGMacros pairs CGM data with Fitbit activity and logged meals (macronutrients and photos) for personalized nutrition research.

- **Participants**: 45 adults (15 normal, 16 prediabetes, 14 type 2 diabetes); no type 1 diabetes and no insulin data
- **Duration**: ~10 days per participant
- **CGM**: Dexcom G6 Pro (5-minute) and FreeStyle Libre Pro (15-minute), worn together
- **License**: CC BY-NC-SA 4.0

## Download Instructions

1. Download the ZIP from https://physionet.org/content/cgmacros/1.0.0/
2. Extract to `cache/data/cgmacros/raw/`

Expected structure:
```
cache/data/cgmacros/raw/
    CGMacros/
        CGMacros-001/
            CGMacros-001.csv
            photos/
        ...
        CGMacros-049/
        bio.csv
```

Participant numbers run to 049 with gaps (024, 025, 037 and 040 are missing).

## Data Files

| File | Description |
|------|-------------|
| CGMacros-XXX.csv | Per-minute CGM (both sensors), Fitbit and meal log, one per participant |
| photos/ | Meal photos, referenced by the `Image path` column |
| bio.csv | Demographics and lab values (not used by the loader) |

## Usage

```python
from src.data.diabetes_datasets import CGMacrosDataLoader

# Load with default settings
loader = CGMacrosDataLoader(use_cached=True)

# Access data
print(f"Patients: {loader.num_patients}")
data = loader.processed_data  # dict[patient_id, DataFrame]

# Load fresh (ignore cache)
loader = CGMacrosDataLoader(use_cached=False)
```

## Output Columns

| Column | Description |
|--------|-------------|
| datetime (index) | Timestamp (5-min intervals) |
| patient_id | Patient ID (`CGMacros-001` → `cgm_1`) |
| bg_mM | Blood glucose (mmol/L), from Dexcom |
| bg_mgdL_dexcom, bg_mgdL_libre | Both sensors (mg/dL): the real reading taken in each 5-min bin, else NaN |
| carbohydrate_g | Meal carbs (g), 0 outside meals |
| msg_type | `ANNOUNCE_MEAL` on meal rows |
| meal_type | breakfast / lunch / dinner / snack |
| meal_calories, protein_g, fat_g, fiber_g | Meal macros (meal rows only) |
| amount_consumed | Raw value; scale is inconsistent (0.75, 100, 300...) |
| image_path | Meal photo path, relative to the participant folder |
| hr_bpm | Heart rate |
| calories, step_count, mets, activity_intensity | Fitbit activity; which ones exist varies by participant |
| cob, iob | Carbs/insulin on board; iob is always NaN (no insulin data) |

## Notes

- The raw CSVs have both sensors linearly interpolated to 1 minute. The loader recovers the real readings (where the interpolated line bends) and drops the interpolated values. Dexcom fills ~90% of rows, Libre ~33% (it reads every 15 min). Most empty Dexcom rows are where only Libre was recording, before Dexcom started or after it ended. The two sensors read at different times.
- Timestamps are date-shifted by the dataset authors for privacy.

## Citation

CGMacros: a scientific dataset for personalized nutrition and diet monitoring (version 1.0.0). PhysioNet. doi:10.13026/3z8q-x658. See the PhysioNet page for the full citation.
