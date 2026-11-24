# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pruner construction (no chronos/GPU required)."""

import pytest

optuna = pytest.importorskip("optuna")

from src.workflows.hpo.engine import build_pruner  # noqa: E402


def test_build_pruner_disabled_returns_none():
    assert build_pruner({"enable_pruning": False}) is None
    assert build_pruner({}) is None


def test_build_pruner_median_default():
    pruner = build_pruner({"enable_pruning": True})
    assert isinstance(pruner, optuna.pruners.MedianPruner)


def test_build_pruner_asha():
    pruner = build_pruner(
        {
            "enable_pruning": True,
            "pruner": {"type": "asha", "min_resource": 200, "reduction_factor": 4},
        }
    )
    assert isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner)


def test_build_pruner_hyperband():
    pruner = build_pruner({"enable_pruning": True, "pruner": {"type": "hyperband"}})
    assert isinstance(pruner, optuna.pruners.HyperbandPruner)


def test_build_pruner_unknown_type_raises():
    with pytest.raises(ValueError):
        build_pruner({"enable_pruning": True, "pruner": {"type": "definitely-typo"}})


def test_append_active_callbacks_within_context_appends_and_calls_original():
    from experiments.metabonet_2026.hpo.chronos2.pruning import (
        active_report_callbacks,
        append_active_callbacks,
    )

    calls = {}

    class FakePipeline:
        def fit(self, *args, callbacks=None, **kwargs):
            calls["callbacks"] = callbacks
            return "fitted"

    sentinel = object()
    pipeline = FakePipeline()
    with active_report_callbacks(sentinel):
        append_active_callbacks(pipeline)
        result = pipeline.fit(inputs=1)
    assert result == "fitted"
    assert calls["callbacks"] == [sentinel]


def test_append_active_callbacks_noop_outside_context():
    from experiments.metabonet_2026.hpo.chronos2.pruning import append_active_callbacks

    calls = {}

    class FakePipeline:
        def fit(self, *args, callbacks=None, **kwargs):
            calls["callbacks"] = callbacks
            return "fitted"

    pipeline = FakePipeline()
    append_active_callbacks(pipeline)  # no active callbacks bound -> no-op
    pipeline.fit(inputs=2)
    assert calls["callbacks"] is None


def test_append_active_callbacks_merges_with_passed_callbacks():
    from experiments.metabonet_2026.hpo.chronos2.pruning import (
        active_report_callbacks,
        append_active_callbacks,
    )

    calls = {}

    class FakePipeline:
        def fit(self, *args, callbacks=None, **kwargs):
            calls["callbacks"] = callbacks
            return "fitted"

    existing, injected = object(), object()
    pipeline = FakePipeline()
    with active_report_callbacks([injected]):
        append_active_callbacks(pipeline)
        pipeline.fit(callbacks=[existing])
    assert calls["callbacks"] == [existing, injected]
