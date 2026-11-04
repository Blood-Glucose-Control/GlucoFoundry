# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Generic evaluation helpers shared across experiments.

Experiment-specific episode construction (e.g. nocturnal midnight anchoring)
lives in the owning recipe under ``experiments/<name>/`` and is resolved via the
experiment registry. This module keeps only experiment-agnostic helpers:

  - :func:`get_holdout_patients` — load a dataset's holdout patient list.
  - :func:`select_episodes_stratified` — reproducible per-patient episode sampling.
"""

from typing import Any, Dict, List

import numpy as np

# Default holdout config directory (can be overridden by callers)
DEFAULT_HOLDOUT_CONFIG_DIR = "configs/data/holdout_10pct"

# Default sampling interval for CGM data (shared by episode builders)
SAMPLING_INTERVAL_MINUTES = 5


def get_holdout_patients(
    dataset: str,
    config_dir: str = DEFAULT_HOLDOUT_CONFIG_DIR,
) -> List[str]:
    """Return the holdout patient list for a dataset."""
    from ..data.versioning.dataset_registry import DatasetRegistry

    hc = DatasetRegistry(holdout_config_dir=config_dir).get_holdout_config(dataset)
    if hc is None or hc.patient_config is None:
        raise ValueError(f"No patient holdout config found for {dataset}")
    return hc.patient_config.holdout_patients


def select_episodes_stratified(
    episodes_by_patient: Dict[str, List[Dict[str, Any]]],
    per_patient: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Stratified episode sampling: pick `per_patient` random episodes from each patient.

    Sorting patients alphabetically before sampling ensures reproducibility
    across calls with the same seed, regardless of dict insertion order.
    """
    rng = np.random.RandomState(seed)
    selected = []
    for pid in sorted(episodes_by_patient.keys()):
        eps = episodes_by_patient[pid]
        n = min(per_patient, len(eps))
        for i in sorted(rng.choice(len(eps), n, replace=False)):
            selected.append(eps[i])
    return selected
