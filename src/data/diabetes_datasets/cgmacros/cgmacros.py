# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)
# For commercial licensing, contact: christopher/cjrisi AT gluroo/uwaterloo DOT com/ca

"""
DataLoader for the CGMacros dataset (PhysioNet v1.0.0).

Study: CGM + Fitbit + macronutrient-annotated meal logs (with photos)
- 45 adults: 15 normal, 16 prediabetes, 14 type 2 diabetes (no T1D, no insulin)
- ~10 days per participant
- Dexcom G6 Pro (5 min) and FreeStyle Libre Pro (15 min) worn together

Data Sources:
- CGMacros/CGMacros-XXX/CGMacros-XXX.csv - per-minute CGM, Fitbit and meal log
"""

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from ...cache_manager import get_cache_manager
from ...dataset_configs import DatasetConfig, get_dataset_config
from ...models import DatasetSourceType
from ..dataset_base import DatasetBase, ProcessedPatientDataFrames
from .data_cleaner import (
    RAW_SUBDIR,
    load_raw_cgmacros_data,
    process_single_patient_data,
)

logger = logging.getLogger(__name__)


class CGMacrosDataLoader(DatasetBase):
    """Data loader for the CGMacros dataset.

    Produces one 5-minute DataFrame per participant with ``bg_mM`` (Dexcom),
    both sensors in mg/dL (``bg_mgdL_dexcom``, ``bg_mgdL_libre``), meal
    events (``carbohydrate_g``, ``msg_type`` plus macros, ``meal_type`` and
    ``image_path``), Fitbit columns, and the derived pipeline features
    (``cob``, ``hour_sin``...). IOB columns are NaN because nobody in the study
    used insulin.

    Attributes:
        keep_columns: Specific columns to load from the dataset.
        use_cached: Whether to use cached processed data if available.
        parallel: Whether to use parallel processing.
        max_workers: Maximum number of workers for parallel processing.

    Example:
        >>> loader = CGMacrosDataLoader(use_cached=True)
        >>> data = loader.processed_data
    """

    def __init__(
        self,
        # Data Selection
        keep_columns: list[str] | None = None,
        # Caching
        use_cached: bool = True,
        # Parallel Processing
        parallel: bool = True,
        max_workers: int = 14,
    ):
        """Initialize the CGMacros data loader.

        Args:
            keep_columns: Optional list of columns to retain per patient.
            use_cached: Whether to load cached processed data when available.
            parallel: Whether patient processing should run in parallel.
            max_workers: Maximum worker count for parallel processing.

        Side Effects:
            Calls load_data() to populate processed_data.
        """
        super().__init__()
        self.use_cached = use_cached
        self.keep_columns = keep_columns
        self.parallel = parallel
        self.max_workers = max_workers

        # Initialize cache manager
        self.cache_manager = get_cache_manager()
        self.dataset_config: DatasetConfig = get_dataset_config(self.dataset_name)

        # Data Objects
        self.raw_data = None
        self.processed_data = None

        # Load data on init
        logger.info(
            "Initializing %s with use_cached=%s.",
            self.__class__.__name__,
            self.use_cached,
        )
        self.load_data()

    # ==================== Properties ====================
    @property
    def dataset_name(self) -> str:
        return DatasetSourceType.CGMACROS.value

    @property
    def description(self) -> str:
        return """
                Title: 'CGMacros: a scientific dataset for personalized nutrition and
                    diet monitoring' (PhysioNet v1.0.0, CC BY-NC-SA 4.0)
                n = 45 adults, ~10 days each
                    - 15 normal, 16 prediabetes, 14 type 2 diabetes
                    - No type 1 diabetes and no insulin data
                    - CGM: Dexcom G6 Pro (5 min, abdomen) + Libre Pro (15 min, arm)
                    - Fitbit heart rate, METs, activity calories
                    - Logged meals with carbs/protein/fat/fiber/calories and photos
                URL: https://physionet.org/content/cgmacros/1.0.0/
            """

    # ==================== Public Methods ====================
    def load_raw(self) -> dict[str, pd.DataFrame]:
        """
        Load raw participant CSVs.

        Returns:
            Dict mapping patient_id -> raw per-minute DataFrame.

        Raises:
            FileNotFoundError: If raw data directory doesn't exist.
        """
        # Ensure raw data exists
        raw_path = self.cache_manager.ensure_raw_data(
            self.dataset_name, self.dataset_config
        )
        return load_raw_cgmacros_data(raw_path / RAW_SUBDIR)

    # ==================== Protected Methods ====================
    def _process_and_cache_data(self) -> ProcessedPatientDataFrames:
        """
        Process raw data and save to cache.
        """
        logger.info("Processing CGMacros raw data...")

        self.processed_data = self._process_raw_data()

        # Save to cache using cache manager's paired save/load methods
        self.cache_manager.save_full_processed_data(
            self.dataset_name, self.processed_data
        )

        logger.info(f"Cached {len(self.processed_data)} patients")
        return self.processed_data

    def _process_raw_data(self) -> ProcessedPatientDataFrames:
        """
        Process raw data into cleaned, per-patient DataFrames.

        Uses parallel processing if self.parallel=True. Participants without
        any glucose readings are skipped.

        Returns:
            Dict mapping patient_id -> DataFrame.
        """
        raw_by_patient = self.load_raw()
        patient_tuples = [
            (patient_id, raw_df) for patient_id, raw_df in raw_by_patient.items()
        ]

        patient_dict: ProcessedPatientDataFrames = {}

        if self.parallel and len(patient_tuples) > 1:
            # Parallel processing
            logger.info(
                f"Processing {len(patient_tuples)} patients in parallel "
                f"(max_workers={self.max_workers})"
            )
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(process_single_patient_data, pt): pt[0]
                    for pt in patient_tuples
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing patients",
                ):
                    patient_id = futures[future]
                    try:
                        pid, patient_df = future.result()
                        if patient_df is not None:
                            patient_dict[pid] = patient_df
                    except Exception as e:
                        logger.error(f"Patient {patient_id} failed: {e}")
        else:
            # Sequential processing (for debugging or single patient)
            logger.info(f"Processing {len(patient_tuples)} patients sequentially")
            for patient_tuple in tqdm(patient_tuples, desc="Processing patients"):
                try:
                    pid, patient_df = process_single_patient_data(patient_tuple)
                    if patient_df is not None:
                        patient_dict[pid] = patient_df
                except Exception as e:
                    logger.error(f"Patient {patient_tuple[0]} failed: {e}")

        if not patient_dict:
            # Don't cache an empty dataset; a later use_cached=True would hide the error
            raise RuntimeError(
                f"All {len(patient_tuples)} CGMacros participants failed to process; "
                "see the errors above."
            )

        logger.info(f"Processed {len(patient_dict)} patients")
        return dict(sorted(patient_dict.items()))
