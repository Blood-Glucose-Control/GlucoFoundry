# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Tests for the nocturnal covariate / model-capability helpers."""

from __future__ import annotations

import pytest

from experiments.nocturnal.covariates import (
    MODEL_PROPERTIES,
    bucket_from_covariates,
    model_supports_past_covariates,
)


# Models whose implementation accepts past covariates (AutoGluon-backed deep /
# gradient-boosted families + covariate-capable foundation models). These are
# gated by ``run_evaluation.py`` before a run starts, so a wrong flag here
# rejects valid ``--covariate-cols`` runs.
@pytest.mark.parametrize("model", ["patchtst", "tsmixer", "tft", "chronos2"])
def test_covariate_capable_models_supported(model: str) -> None:
    assert model_supports_past_covariates(model) is True


@pytest.mark.parametrize("model", ["naive_baseline", "unknown_model_xyz"])
def test_non_covariate_models_not_supported(model: str) -> None:
    # Unknown / bare keys without a registry match default to False.
    assert model_supports_past_covariates(model) is False


def test_supports_past_covariates_matches_base_name_variant() -> None:
    # Keys may carry a ``parent/variant`` suffix; an unmatched full name falls
    # back to the base model (e.g. ``chronos2/v2`` -> ``chronos2``).
    assert (
        model_supports_past_covariates("chronos2/v2")
        == (MODEL_PROPERTIES["chronos2"]["supports_past_covariates"])
    )


def test_patchtst_registry_flags_match_implementation() -> None:
    # Regression guard: PatchTST is AutoGluon-backed with covariate + quantile
    # support (see src/models/patchtst/{config,model}.py), so its registry flags
    # must reflect that.
    entry = MODEL_PROPERTIES["patchtst"]
    assert entry["supports_past_covariates"] is True
    assert entry["probabilistic"] is True
    assert entry["univariate_only"] is False


@pytest.mark.parametrize(
    ("mode", "cov_cols", "expected"),
    [
        ("zeroshot", None, "zero_shot"),
        ("finetuned", [], "bg_only"),
        ("finetuned", ["iob"], "iob"),
        ("finetuned", ["iob", "cob"], "iob_cob"),
        ("finetuned", ["insulin_availability", "carbs"], "iob_cob"),
    ],
)
def test_bucket_from_covariates(mode: str, cov_cols, expected: str) -> None:
    assert bucket_from_covariates(mode, cov_cols) == expected
