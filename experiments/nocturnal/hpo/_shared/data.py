# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Train/holdout data loading for the nocturnal HPO sweeps.

Loads the per-dataset train/holdout splits via the core
:class:`~src.data.versioning.dataset_registry.DatasetRegistry` and concatenates
them into a single train frame (for per-trial fitting) and a single holdout
frame (for midnight-anchored evaluation). Patient IDs are already standardized
to the ``patient_id`` column by the dataset loaders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, cast

import pandas as pd

from src.data.versioning.dataset_registry import DatasetRegistry

logger = logging.getLogger(__name__)

PATIENT_COL = "patient_id"

DEFAULT_DATASETS: Sequence[str] = ("aleppo_2017", "brown_2019", "lynch_2022")
DEFAULT_HOLDOUT_CONFIG_DIR = "configs/data/holdout_10pct"


@dataclass
class NocturnalData:
    """Combined train/holdout frames for a nocturnal sweep."""

    train: pd.DataFrame
    holdout: pd.DataFrame
    datasets: Sequence[str]

    @property
    def n_holdout_patients(self) -> int:
        return int(self.holdout[PATIENT_COL].nunique())


def build_data(config: Dict[str, Any]) -> NocturnalData:
    """Load and concatenate train/holdout splits for the configured datasets."""
    data_cfg: Dict[str, Any] = config.get("data", {}) or {}
    datasets: Sequence[str] = data_cfg.get("datasets", DEFAULT_DATASETS)
    holdout_config_dir: str = data_cfg.get(
        "holdout_config_dir", DEFAULT_HOLDOUT_CONFIG_DIR
    )
    patient_limit: Optional[int] = data_cfg.get("patient_limit")

    registry = DatasetRegistry(holdout_config_dir=holdout_config_dir)
    train_frames: List[pd.DataFrame] = []
    holdout_frames: List[pd.DataFrame] = []
    for dataset in datasets:
        train_df, holdout_df = registry.load_dataset_with_split(dataset)
        train_frames.append(train_df)
        holdout_frames.append(holdout_df)
        logger.info(
            "Loaded %s: train=%d rows, holdout=%d rows",
            dataset,
            len(train_df),
            len(holdout_df),
        )

    train = pd.concat(train_frames, ignore_index=True)
    holdout = pd.concat(holdout_frames, ignore_index=True)

    if patient_limit:
        # Smoke mode: restrict to a handful of holdout patients (and their train
        # rows) so a sweep can be exercised end-to-end quickly.
        keep = list(holdout[PATIENT_COL].unique())[: int(patient_limit)]
        holdout = holdout[holdout[PATIENT_COL].isin(keep)].copy()
        train = train[train[PATIENT_COL].isin(keep)].copy()
        logger.info("patient_limit=%s -> %d holdout patients", patient_limit, len(keep))

    return NocturnalData(
        train=cast(pd.DataFrame, train),
        holdout=cast(pd.DataFrame, holdout),
        datasets=list(datasets),
    )
