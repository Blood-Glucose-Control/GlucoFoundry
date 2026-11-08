# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Tests for the experiment registry and auto-discovery.

These exercise the additive registry scaffolding (no experiment code is moved
yet): registration, lookup, duplicate handling, and path-based auto-discovery of
``experiments/<name>/register.py`` recipes.
"""

import textwrap

import pytest

from src.experiments.registry import (
    ARTIFACTS_ROOT_ENV,
    ExperimentPaths,
    ExperimentSpec,
    available_experiments,
    discover_experiments,
    experiments_root,
    get_episode_builder,
    get_experiment,
    get_experiment_paths,
    register_experiment,
    reset_registry,
)


@pytest.fixture(autouse=True)
def _clean_registry(tmp_path_factory):
    reset_registry()
    # Isolate these unit tests from implicit auto-discovery of the real
    # experiments/ recipes so assertions are deterministic. Tests that exercise
    # discovery call discover_experiments(root=..., force=True) explicitly.
    discover_experiments(root=tmp_path_factory.mktemp("empty_experiments"), force=True)
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


def test_get_episode_builder_returns_callable():
    def build(_data):
        return ["episode"]

    register_experiment(ExperimentSpec(name="demo", build_episodes=build))
    builder = get_episode_builder("demo")
    assert builder(None) == ["episode"]


def test_get_episode_builder_raises_when_absent():
    register_experiment(ExperimentSpec(name="demo"))  # no build_episodes
    with pytest.raises(ValueError, match="no episode builder"):
        get_episode_builder("demo")


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


class TestExperimentPaths:
    """Canonical experiment-owned path resolution (RFC-001)."""

    def test_default_paths_are_repo_local(self):
        paths = get_experiment_paths("nocturnal")
        root = experiments_root() / "nocturnal"
        assert isinstance(paths, ExperimentPaths)
        assert paths.name == "nocturnal"
        assert paths.root == root
        assert paths.configs == root / "configs"
        assert paths.scripts == root / "scripts"
        assert paths.docs == root / "docs"
        assert paths.results == root / "results"
        assert paths.trained_models == root / "trained_models"

    def test_artifacts_root_override_param(self, tmp_path):
        paths = get_experiment_paths("demo", artifacts_root=tmp_path)
        # Large artifacts redirect under <override>/<name>/ ...
        assert paths.results == tmp_path / "demo" / "results"
        assert paths.trained_models == tmp_path / "demo" / "trained_models"
        # ... but versioned configs stay repo-local.
        assert paths.configs == experiments_root() / "demo" / "configs"

    def test_artifacts_root_override_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ARTIFACTS_ROOT_ENV, str(tmp_path))
        paths = get_experiment_paths("demo")
        assert paths.results == tmp_path / "demo" / "results"
        assert paths.configs == experiments_root() / "demo" / "configs"

    def test_param_override_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ARTIFACTS_ROOT_ENV, str(tmp_path / "env"))
        paths = get_experiment_paths("demo", artifacts_root=tmp_path / "param")
        assert paths.results == tmp_path / "param" / "demo" / "results"

    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_invalid_name_raises(self, bad):
        with pytest.raises(ValueError):
            get_experiment_paths(bad)  # type: ignore[arg-type]

    def test_ensure_creates_artifact_dirs(self, tmp_path):
        paths = get_experiment_paths("demo", artifacts_root=tmp_path)
        assert not paths.results.exists()
        returned = paths.ensure()
        assert returned is paths
        assert paths.results.is_dir()
        assert paths.trained_models.is_dir()
