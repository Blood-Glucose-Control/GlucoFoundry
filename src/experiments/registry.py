# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Experiment registry with auto-discovery.

Experiments are self-contained recipes under the top-level ``experiments/``
directory. Each recipe provides an ``experiments/<name>/register.py`` module that
calls :func:`register_experiment` with an :class:`ExperimentSpec`.

The core never imports experiment modules directly; it resolves them by name
through this registry. An experiment can therefore be dropped simply by deleting
its directory, with no edits to core code (see ``docs/design/toolkit-charter.md``
§3.2–3.3).

This module is additive scaffolding: it introduces the registry and discovery
mechanism without yet moving experiment code or removing existing imports.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Name of the per-experiment registration module scanned under experiments/<name>/.
REGISTER_MODULE_FILENAME = "register.py"


@dataclass(frozen=True)
class ExperimentSpec:
    """Declarative description of an experiment recipe.

    All hooks are optional so a recipe can opt into only what it provides. The
    core resolves these by experiment name rather than importing the experiment
    package.

    Attributes:
        name: Unique experiment identifier (e.g. ``"nocturnal"``).
        build_episodes: Callable that constructs evaluation episodes for this
            experiment's protocol (e.g. midnight-anchored episodes).
        summarizer: Summarizer class for this experiment's results.
        evaluate: Evaluation entry point for this experiment.
        description: Short human-readable description.
    """

    name: str
    build_episodes: Callable[..., Any] | None = None
    summarizer: type | None = None
    evaluate: Callable[..., Any] | None = None
    description: str = ""


_REGISTRY: dict[str, ExperimentSpec] = {}
# Re-entrant: discovery holds the lock while loading register modules, which call
# register_experiment() on the same thread.
_LOCK = threading.RLock()
_discovered = False


def register_experiment(spec: ExperimentSpec) -> None:
    """Register an experiment recipe.

    Idempotent for the same spec object; raises on a conflicting re-registration
    of the same name.
    """
    if not spec.name:
        raise ValueError("ExperimentSpec.name must be a non-empty string")
    with _LOCK:
        existing = _REGISTRY.get(spec.name)
        if existing is not None and existing is not spec:
            raise ValueError(
                f"Experiment {spec.name!r} is already registered to a different spec"
            )
        _REGISTRY[spec.name] = spec


def get_experiment(name: str) -> ExperimentSpec:
    """Return the registered spec for ``name`` (triggering discovery if needed)."""
    _ensure_discovered()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"No experiment registered under {name!r}. "
            f"Available: {available_experiments()}"
        ) from None


def available_experiments() -> list[str]:
    """Return the sorted names of all registered experiments."""
    _ensure_discovered()
    with _LOCK:
        return sorted(_REGISTRY)


def get_episode_builder(name: str) -> Callable[..., Any]:
    """Return the episode builder for experiment ``name``.

    Raises if the experiment is unknown or declares no ``build_episodes`` hook,
    so callers get a guaranteed non-optional callable.
    """
    spec = get_experiment(name)
    if spec.build_episodes is None:
        raise ValueError(f"Experiment {name!r} declares no episode builder")
    return spec.build_episodes


def experiments_root() -> Path:
    """Absolute path to the top-level ``experiments/`` recipes directory.

    ``src/experiments/registry.py`` -> ``parents[2]`` is the repository root.
    """
    return Path(__file__).resolve().parents[2] / "experiments"


def discover_experiments(root: Path | None = None, *, force: bool = False) -> None:
    """Scan ``experiments/*/register.py`` and import each to trigger registration.

    Args:
        root: Recipes directory to scan. Defaults to :func:`experiments_root`.
        force: Re-run discovery even if it has already completed.
    """
    global _discovered
    with _LOCK:
        if _discovered and not force:
            return
        search = root if root is not None else experiments_root()
        if search.is_dir():
            for register_py in sorted(search.glob(f"*/{REGISTER_MODULE_FILENAME}")):
                _load_register_module(register_py)
        _discovered = True


def reset_registry() -> None:
    """Clear the registry and discovery state. Intended for tests."""
    global _discovered
    with _LOCK:
        _REGISTRY.clear()
        _discovered = False


def _ensure_discovered() -> None:
    if not _discovered:
        discover_experiments()


def _load_register_module(path: Path) -> None:
    module_name = f"_glucofoundry_experiment_{path.parent.name}_register"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            logger.warning("Could not create import spec for %s", path)
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception:
        # A broken experiment recipe must not take down the whole toolkit.
        logger.exception("Failed to load experiment registration at %s", path)
