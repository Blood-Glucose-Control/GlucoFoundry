"""Reusable, experiment-agnostic hyperparameter-optimization engine.

:mod:`.engine` provides the :class:`~.engine.StudySpec` contract and the shared
Optuna study orchestration (storage, sampler, pruner, early-stopping, warm-start).
Experiment-specific sweep trees supply a ``StudySpec`` and live with their
experiment recipe (for example :mod:`experiments.metabonet_2026.hpo`).
"""
