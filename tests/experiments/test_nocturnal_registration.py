# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Verify the nocturnal recipe registers itself via auto-discovery.

Exercises the end-to-end contract: discovering ``experiments/*/register.py``
loads the nocturnal recipe and registers its hooks, so the core can resolve it by
name without importing the recipe directly.
"""

import pytest

from src.experiments.registry import (
    available_experiments,
    discover_experiments,
    get_experiment,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def test_nocturnal_recipe_is_discovered_and_registered():
    discover_experiments(force=True)

    assert "nocturnal" in available_experiments()

    spec = get_experiment("nocturnal")
    assert spec.name == "nocturnal"
    assert spec.build_episodes is not None
    assert spec.summarizer is not None
    assert spec.evaluate is not None
    assert "hypoglycemia" in spec.description.lower()
