# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

import pandas as pd


def mg_dl_to_mmol_l(df: pd.DataFrame, bgl_col: str = "bg-0:00") -> pd.Series:
    return (df[bgl_col] / 18.018).round(2)
