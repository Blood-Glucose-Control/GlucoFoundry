# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Neutral, model-agnostic Optuna sweep engine.

Shared study infrastructure for per-model-family sweeps (which live with their
experiment, e.g. ``experiments/metabonet_2026/hpo/<family>/``): parallel-safe
storage selection, sampler and pruner selection, objective parsing, the
``create``/``run`` command loop, and the worker CLI. A family (or experiment)
supplies only a :class:`StudySpec` — its default config, search-space bounds
builder, objective factory, data builder, and an optional dashboard-metadata
hook — and calls :func:`run_main`.

Previously this engine lived inside the ``chronos2`` family package and
every other family imported from it; it now stands alone so no family is the
load-bearing home for the generic machinery.
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import optuna
import yaml

logger = logging.getLogger(__name__)


# A family-provided objective factory:
#   (config, data, bounds, gpu_label, metric_names, is_multi) -> objective callable
ObjectiveFactory = Callable[[Dict[str, Any], Any, Any, str, Sequence[str], bool], Any]


@dataclass(frozen=True)
class StudySpec:
    """Everything a model family contributes to a sweep.

    Attributes:
        default_config: The family's ``DEFAULT_CONFIG`` dict (study name, storage,
            data block, search_space, smoke overrides, ...).
        build_bounds: ``(config) -> bounds`` search-space object for the family.
        build_objective: ``(config, data, bounds, gpu_label, metric_names,
            is_multi) -> objective`` factory constructing the family objective.
        build_data: ``(config) -> data`` builder (currently the shared MetaboNet
            pipeline; experiment-specific data extraction is tracked in #79).
        apply_dashboard_metadata: Optional ``(study, config) -> None`` hook that
            attaches family-specific human-readable attributes for the dashboard.
        catch_trial_exceptions: When True (default), a failing trial is marked
            FAILED and the worker continues rather than tearing down.
    """

    default_config: Dict[str, Any]
    build_bounds: Callable[[Dict[str, Any]], Any]
    build_objective: ObjectiveFactory
    build_data: Callable[[Dict[str, Any]], Any]
    apply_dashboard_metadata: Optional[
        Callable[[optuna.Study, Dict[str, Any]], None]
    ] = None
    catch_trial_exceptions: bool = True


# --- generic helpers ---


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_objectives(config: Dict[str, Any]):
    """Return (metric_names, directions). Multi-objective iff len > 1.

    Uses the ``objectives`` list (each: {metric, direction}) when present,
    otherwise falls back to the single ``primary_metric`` + ``direction``.
    """
    objectives = config.get("objectives")
    if objectives:
        metrics = [str(o["metric"]) for o in objectives]
        directions = [str(o.get("direction", "minimize")) for o in objectives]
        return metrics, directions
    return [config.get("primary_metric", "rmse_120")], [
        config.get("direction", "minimize")
    ]


def _ensure_storage_dir(storage: str) -> None:
    """Create the parent directory for a sqlite:/// storage path if needed."""
    prefix = "sqlite:///"
    if storage.startswith(prefix):
        db_path = Path(storage[len(prefix) :])
        db_path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_journal_path(storage: str) -> Path:
    """Map the configured ``storage`` location to a JournalStorage log file path.

    Accepts a plain filesystem path, a ``journal:///`` URL, or a legacy
    ``sqlite:///foo.db`` URL (mapped to a sibling ``foo.journal`` so existing
    configs keep working under the journal default).
    """
    if storage.startswith("sqlite:///"):
        return Path(storage[len("sqlite:///") :]).with_suffix(".journal")
    if storage.startswith("journal:///"):
        return Path(storage[len("journal:///") :])
    return Path(storage)


def _build_journal_storage(storage: str):
    """Parallel-safe, file-locked ``JournalStorage`` (the default backend).

    Optuna recommends journal file storage over SQLite for many cooperative
    workers because SQLite's write lock serializes concurrent trial writes.
    Trade-off: ``JournalStorage`` has no heartbeat/stale-trial retry, so a
    crashed worker's trial stays RUNNING until manually cleaned — use the
    ``rdb`` backend when crashed-worker auto-retry matters more than write
    concurrency.
    """
    from optuna.storages import JournalStorage
    from optuna.storages.journal import JournalFileBackend

    path = _resolve_journal_path(storage)
    path.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(path)))


def _build_rdb_storage(config: Dict[str, Any]):
    """Build an RDB storage with heartbeat so crashed trials auto-fail/retry.

    Without a heartbeat, Optuna cannot tell a dead worker's trial from a slow
    one, so killed-worker trials linger forever as RUNNING. Heartbeat lets any
    live worker mark stale trials FAILED (grace_period after the last beat) and
    RetryHeartbeatStaleTrialCallback re-enqueues them. Heartbeat is an RDB-only
    feature; against many parallel workers prefer a real RDB server (Postgres/
    MySQL) over SQLite to avoid write-lock contention.
    """
    from optuna.storages import RDBStorage, RetryHeartbeatStaleTrialCallback

    url = config["storage"]
    _ensure_storage_dir(url)
    heartbeat_interval = int(config.get("heartbeat_interval", 60))
    grace_period = int(config.get("grace_period", 300))
    max_retry = config.get("max_retry", 2)
    return RDBStorage(
        url=url,
        heartbeat_interval=heartbeat_interval,
        grace_period=grace_period,
        heartbeat_stale_trial_callback=RetryHeartbeatStaleTrialCallback(
            max_retry=max_retry
        ),
    )


def _build_storage(config: Dict[str, Any]):
    """Select the Optuna storage backend from config.

    ``storage_backend`` is ``"journal"`` by default (parallel-safe file lock);
    set it to ``"rdb"`` for the heartbeat/auto-retry RDB backend. ``storage``
    is the location: an RDB URL for ``rdb`` (e.g. ``sqlite:///x.db``), or a
    journal log path for ``journal`` (a ``sqlite:///x.db`` URL maps to
    ``x.journal``).
    """
    backend = str(config.get("storage_backend", "journal")).lower()
    if backend in ("journal", "file"):
        return _build_journal_storage(config["storage"])
    if backend in ("rdb", "sqlite", "relational"):
        return _build_rdb_storage(config)
    raise ValueError(
        f"Unknown storage_backend {config.get('storage_backend')!r}; use "
        "'journal' (parallel-safe, default) or 'rdb' (heartbeat/auto-retry)."
    )


def _build_sampler(config: Dict[str, Any]):
    """Build the Optuna sampler from config (default multivariate TPE).

    Options: "tpe" (default), "nsga2" (genetic/population-based; usable for
    single-objective and the natural choice if we go multi-objective), "random",
    or "cmaes". TPE is the strong default for our large mixed/conditional space.
    """
    spec = config.get("sampler", {}) or {}
    kind = str(spec.get("type", "tpe")).lower()
    seed = spec.get("seed")
    if seed is not None:
        # CRITICAL for distributed runs: workers must NOT share a seed, or they
        # sample identical initial trials (and stay synchronized for several
        # generations), wasting compute. Offset by a per-process value so each
        # worker's sampler diverges while a base seed still aids debuggability.
        seed = int(seed) + (os.getpid() % 100_000)
    if kind == "nsga2":
        return optuna.samplers.NSGAIISampler(
            population_size=spec.get("population_size", 50), seed=seed
        )
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if kind == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    return optuna.samplers.TPESampler(multivariate=True, group=True, seed=seed)


def _make_early_stopping_callback(patience: int):
    """Callback that stops the worker if best value stalls for `patience` trials."""
    state = {"best": None, "since_improve": 0}

    def _callback(study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
        try:
            current = study.best_value
        except ValueError:
            return
        if state["best"] is None or current < state["best"] - 1e-9:
            state["best"] = current
            state["since_improve"] = 0
        else:
            state["since_improve"] += 1
            if state["since_improve"] >= patience:
                logger.info(
                    "Early stopping: no improvement in %d trials (best=%.4f).",
                    patience,
                    state["best"],
                )
                study.stop()

    return _callback


def build_pruner(config: dict):
    """Construct an Optuna pruner from a config dict, or None if disabled."""
    import optuna

    if not config.get("enable_pruning"):
        return None
    spec = config.get("pruner", {}) or {}
    kind = str(spec.get("type", "median")).lower()
    if kind in ("asha", "successive_halving", "successivehalving"):
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=spec.get("min_resource", 100),
            reduction_factor=spec.get("reduction_factor", 3),
            min_early_stopping_rate=spec.get("min_early_stopping_rate", 0),
        )
    if kind == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=spec.get("min_resource", 100),
            max_resource=spec.get("max_resource", "auto"),
            reduction_factor=spec.get("reduction_factor", 3),
        )
    if kind == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=spec.get("n_startup_trials", 5),
            n_warmup_steps=spec.get("n_warmup_steps", 0),
        )
    raise ValueError(
        f"Unknown pruner type {spec.get('type')!r}; use 'median', 'asha', "
        f"or 'hyperband'."
    )


def _warm_start(study: optuna.Study, config: Dict[str, Any]) -> None:
    """Seed a fresh single-objective study with a prior study's trials."""
    source = config.get("warm_start_from")
    if not source or len(study.trials) > 0:
        return
    primary = _parse_objectives(config)[0][0]
    src = optuna.load_study(study_name=str(source), storage=_build_storage(config))
    added = 0
    for t in src.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        value = t.user_attrs.get(primary)
        if value is None or not np.isfinite(value):
            continue
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                params=t.params,
                distributions=t.distributions,
                value=float(value),
            )
        )
        added += 1
    print(f"Warm-started {added} trials from '{source}' (metric={primary}).")


def load_config(
    spec: StudySpec, path: Optional[str], smoke: bool = False
) -> Dict[str, Any]:
    """Load a run config: the family default, layered with YAML + smoke overrides."""
    config = copy.deepcopy(spec.default_config)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            config = _deep_merge(config, yaml.safe_load(fh) or {})
    if smoke:
        config = _deep_merge(config, config.get("smoke", {}))
    return config


def create_study(spec: StudySpec, config: Dict[str, Any]) -> optuna.Study:
    """Create (or load) the shared study and attach dashboard metadata."""
    metric_names, directions = _parse_objectives(config)
    is_multi = len(metric_names) > 1
    create_kwargs: Dict[str, Any] = dict(
        study_name=config["study_name"],
        storage=_build_storage(config),
        load_if_exists=True,
        sampler=_build_sampler(config),
    )
    if is_multi:
        # Optuna cannot prune multi-objective studies; ignore any pruner config.
        create_kwargs["directions"] = directions
    else:
        create_kwargs["direction"] = directions[0]
        create_kwargs["pruner"] = build_pruner(config)
    study = optuna.create_study(**create_kwargs)
    try:
        study.set_metric_names(list(metric_names))
    except Exception:  # noqa: BLE001 - older storages may not support it
        pass
    if spec.apply_dashboard_metadata is not None:
        spec.apply_dashboard_metadata(study, config)
    return study


def cmd_create(spec: StudySpec, config: Dict[str, Any]) -> int:
    study = create_study(spec, config)
    _warm_start(study, config)
    backend = str(config.get("storage_backend", "journal")).lower()
    location = (
        _resolve_journal_path(config["storage"])
        if backend in ("journal", "file")
        else config["storage"]
    )
    print(f"Study '{study.study_name}' ready ({backend} storage at {location})")
    print(f"Existing trials: {len(study.trials)}")
    return 0


def cmd_run(
    spec: StudySpec, config: Dict[str, Any], n_trials: int, gpu_label: str
) -> int:
    data = spec.build_data(config)
    bounds = spec.build_bounds(config)
    metric_names, _ = _parse_objectives(config)
    is_multi = len(metric_names) > 1
    objective = spec.build_objective(
        config,
        data,
        bounds,
        gpu_label or os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        metric_names,
        is_multi,
    )
    study = create_study(spec, config)

    timeout = config.get("timeout")
    # n_trials <= 0 means "unlimited until timeout" (requires a timeout).
    effective_n_trials: Optional[int] = n_trials if n_trials and n_trials > 0 else None
    if effective_n_trials is None and timeout is None:
        raise ValueError(
            "Set a positive --n-trials or a config 'timeout' so the worker terminates."
        )

    callbacks = []
    patience = config.get("early_stopping_patience")
    if patience and not is_multi:
        # study.best_value is undefined for multi-objective; skip the callback.
        callbacks.append(_make_early_stopping_callback(int(patience)))

    optimize_kwargs: Dict[str, Any] = dict(
        n_trials=effective_n_trials,
        timeout=timeout,
        gc_after_trial=True,
        callbacks=callbacks,
    )
    if spec.catch_trial_exceptions:
        # Resilience: a single failing trial should be marked FAILED and let the
        # worker keep going. Optuna logs each failure at WARNING level.
        optimize_kwargs["catch"] = (Exception,)
    study.optimize(objective, **optimize_kwargs)

    print(f"Worker finished. Study now has {len(study.trials)} trials.")
    if is_multi:
        front = study.best_trials
        print(f"Pareto front ({', '.join(metric_names)}): {len(front)} trials")
        for t in front[:5]:
            vals = ", ".join(f"{m}={v:.3f}" for m, v in zip(metric_names, t.values))
            print(f"  trial {t.number}: {vals}")
    else:
        try:
            best = study.best_trial
            print(f"Best {metric_names[0]}: {best.value:.3f} (trial {best.number})")
        except ValueError:
            print("No completed trials yet.")
    return 0


def build_parser(description: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["create", "run"], help="Action to run.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path.")
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--gpu-label", type=str, default="")
    parser.add_argument("--study-name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument(
        "--storage-backend",
        type=str,
        choices=["journal", "rdb"],
        default=None,
        help="Optuna storage backend (default: journal, parallel-safe).",
    )
    parser.add_argument("--patient-limit", type=int, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--keep-models", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="Use tiny smoke-test settings."
    )
    return parser


def run_main(
    spec: StudySpec,
    argv: Optional[Sequence[str]] = None,
    description: Optional[str] = None,
) -> int:
    """Generic worker entry point: parse args, apply CLI overrides, dispatch.

    Pass the calling family module's ``__doc__`` as ``description`` so ``--help``
    shows that family's usage text (with its formatting preserved).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    args = build_parser(description).parse_args(argv)
    config = load_config(spec, args.config, smoke=args.smoke)

    # CLI overrides take precedence over config/defaults.
    if args.study_name:
        config["study_name"] = args.study_name
    if args.storage:
        config["storage"] = args.storage
    if args.storage_backend:
        config["storage_backend"] = args.storage_backend
    if args.output_root:
        config["output_root"] = args.output_root
    if args.keep_models:
        config["keep_models"] = True
    if args.patient_limit is not None:
        config.setdefault("data", {})["patient_limit"] = args.patient_limit

    if args.command == "create":
        return cmd_create(spec, config)
    return cmd_run(spec, config, n_trials=args.n_trials, gpu_label=args.gpu_label)
