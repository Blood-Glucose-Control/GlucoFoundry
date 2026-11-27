import numpy as np
import pandas as pd

from src.data.diabetes_datasets.tamborlane_2008.data_cleaner import (
    process_single_patient_tamborlane,
)


def _synthetic_patient(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
            "patient_id": ["tam_001"] * n,
            "bg_mM": np.linspace(5.0, 9.0, n),
            "msg_type": [""] * n,
        }
    )


def test_process_single_patient_sets_datetime_as_index_only():
    """Regression for the 'datetime is both an index level and a column label'
    failure: the cleaner must set datetime as the index *only* (drop the column),
    like every other dataset cleaner, so the downstream merge_asof on "datetime"
    in ensure_regular_time_intervals() is unambiguous.
    """
    df = _synthetic_patient()

    pid, processed = process_single_patient_tamborlane(
        ("tam_001", df, pd.Timestamp("2020-01-01"))
    )

    assert pid == "tam_001"
    assert not processed.empty
    assert isinstance(processed.index, pd.DatetimeIndex)
    assert processed.index.name == "datetime"
    # datetime must live only on the index, never also as a column.
    assert "datetime" not in processed.columns
    assert "patient_id" in processed.columns
