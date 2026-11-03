# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Tests for the experiment registry and auto-discovery.

These exercise the additive registry scaffolding (no experiment code is moved
yet): registration, lookup, duplicate handling, and path-based auto-discovery of
``experiments/<name>/register.py`` recipes.
"""

import textwrap

import pytest

from src.experiments.registry import (
    ExperimentSpec,
    available_experiments,
    discover_experiments,
    get_experiment,
    register_experiment,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_registry()
    yield
    reset_registry()


def test_register_and_get():
    spec = ExperimentSpec(name="demo", description="a demo experiment")
    register_experiment(spec)
    assert get_experiment("demo") is spec
    assert "demo" in available_experiments()


def test_register_same_spec_is_idempotent():
    spec = ExperimentSpec(name="demo")
    register_experiment(spec)
    register_experiment(spec)  # no error for the same object
    assert available_experiments() == ["demo"]


def test_conflicting_registration_raises():
    register_experiment(ExperimentSpec(name="demo"))
    with pytest.raises(ValueError, match="already registered"):
        register_experiment(ExperimentSpec(name="demo", description="different"))


def test_empty_name_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        register_experiment(ExperimentSpec(name=""))


def test_get_unknown_experiment_raises():
    with pytest.raises(KeyError, match="No experiment registered"):
        get_experiment("missing")


def test_hooks_are_resolvable_by_name():
    def build(_data):
        return ["episode"]

    register_experiment(ExperimentSpec(name="demo", build_episodes=build))
    resolved = get_experiment("demo")
    assert resolved.build_episodes is not None
    assert resolved.build_episodes(None) == ["episode"]


def test_auto_discovery_loads_register_modules(tmp_path):
    recipe = tmp_path / "sample" / "register.py"
    recipe.parent.mkdir(parents=True)
    recipe.write_text(
        textwrap.dedent(
            """
            from src.experiments.registry import ExperimentSpec, register_experiment

            register_experiment(ExperimentSpec(name="sample", description="discovered"))
            """
        )
    )

    discover_experiments(root=tmp_path, force=True)

    assert "sample" in available_experiments()
    assert get_experiment("sample").description == "discovered"


def test_discovery_skips_dirs_without_register(tmp_path):
    (tmp_path / "no_register").mkdir()
    discover_experiments(root=tmp_path, force=True)
    assert available_experiments() == []


def test_broken_recipe_does_not_break_discovery(tmp_path):
    good = tmp_path / "good" / "register.py"
    good.parent.mkdir(parents=True)
    good.write_text(
        "from src.experiments.registry import ExperimentSpec, register_experiment\n"
        "register_experiment(ExperimentSpec(name='good'))\n"
    )
    broken = tmp_path / "broken" / "register.py"
    broken.parent.mkdir(parents=True)
    broken.write_text("raise RuntimeError('boom')\n")

    discover_experiments(root=tmp_path, force=True)

    # The good recipe still registers despite the broken one.
    assert "good" in available_experiments()
    assert "broken" not in available_experiments()
