# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Optuna pruning/reporting callbacks for Chronos-2 fine-tuning.

Builds the HuggingFace ``TrainerCallback``s used during Chronos-2 fine-tuning
(an Optuna pruning/reporting callback, early stopping, a step recorder) and the
plumbing that forwards them to the Trainer **without** globally patching
``Chronos2Pipeline.fit``:

- :class:`active_report_callbacks` binds the callbacks for the current fit.
- :func:`append_active_callbacks` wraps a pipeline's (instance) ``fit`` to append
  them, used by
  :class:`experiments.metabonet_2026.hpo.chronos2.reporting.ReportingChronos2Model`
  (a scoped ``Chronos2Model`` subclass).

The callback builders import ``transformers`` lazily, so this module stays
importable without the model dependencies installed.
"""

from __future__ import annotations

import contextvars
from typing import List, Optional

# Extra HF Trainer callbacks bound for the fine-tune currently running in this
# worker process (pruning/reporting, early stopping, step telemetry).
_ACTIVE_CALLBACKS: contextvars.ContextVar[Optional[List[object]]] = (
    contextvars.ContextVar("chronos2_active_callbacks", default=None)
)


class active_report_callbacks:
    """Context manager binding extra Trainer callbacks for the wrapped fit.

    Accepts a single callback or a list; :func:`append_active_callbacks` appends
    them to fits started inside the ``with`` block.
    """

    def __init__(self, callbacks) -> None:
        if callbacks is None:
            self.callbacks: List[object] = []
        elif isinstance(callbacks, (list, tuple)):
            self.callbacks = list(callbacks)
        else:
            self.callbacks = [callbacks]
        self._token = None

    def __enter__(self) -> List[object]:
        self._token = _ACTIVE_CALLBACKS.set(self.callbacks)
        return self.callbacks

    def __exit__(self, *exc) -> bool:
        if self._token is not None:
            _ACTIVE_CALLBACKS.reset(self._token)
        return False


def append_active_callbacks(pipeline) -> None:
    """Wrap ``pipeline.fit`` (in place, idempotently) to append whatever
    callbacks are bound via :class:`active_report_callbacks` **at fit time**.

    The wrapper consults the ContextVar on each call, so a fit started after the
    ``active_report_callbacks`` block exits receives no extra callbacks. Wrapping
    is idempotent -- an already-wrapped ``fit`` is left untouched -- so repeated
    ``load_model_pipeline()`` calls can't stack duplicate appends.
    ``Chronos2Pipeline.fit`` returns a fresh fine-tuned pipeline, so the
    instance-level wrap never reaches the saved model.
    """
    original_fit = pipeline.fit
    if getattr(original_fit, "_appends_active_callbacks", False):
        return

    def _wrapped_fit(*args, **kwargs):
        active = _ACTIVE_CALLBACKS.get()
        if active:
            callbacks = list(kwargs.get("callbacks") or [])
            callbacks.extend(active)
            kwargs["callbacks"] = callbacks
        return original_fit(*args, **kwargs)

    _wrapped_fit._appends_active_callbacks = True  # type: ignore[attr-defined]
    pipeline.fit = _wrapped_fit


def make_early_stopping_callback(patience: int):
    """HF ``EarlyStoppingCallback`` that stops fine-tuning once the validation
    metric (``eval_loss``, best-checkpoint metric) has not improved for
    ``patience`` evaluations. Requires ``load_best_model_at_end`` (set by the
    Chronos-2 pipeline when ``eval_during_fine_tune`` is on).
    """
    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(early_stopping_patience=int(patience))


def make_step_recorder_callback():
    """Return ``(callback, holder)`` where ``holder['steps']`` is updated to the
    latest HF ``global_step`` so telemetry reflects the ACTUAL steps trained
    (the surviving best checkpoint alone under-reports it when
    ``save_total_limit`` deletes later checkpoints).
    """
    from transformers.trainer_callback import TrainerCallback

    holder: dict[str, Optional[int]] = {"steps": None}

    class _StepRecorder(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            holder["steps"] = int(state.global_step)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            holder["steps"] = int(state.global_step)
            return control

    return _StepRecorder(), holder


def make_pruning_callback(
    trial, report_metric: str = "eval_loss", *, prune: bool = True
):
    """Build a HuggingFace ``TrainerCallback`` that reports to Optuna.

    Always reports the validation metric as an Optuna intermediate value (so the
    dashboard shows per-trial training curves). When ``prune`` is True it also
    stops the trial early if ``trial.should_prune()``; when False it is a
    reporting-only callback (used for single-objective runs that want training
    curves without pruning). NOTE: this must never be attached for
    multi-objective studies — Optuna's ``trial.report()`` raises
    ``NotImplementedError`` for them.
    """
    import optuna
    from transformers.trainer_callback import TrainerCallback

    class OptunaPruningCallback(TrainerCallback):
        def __init__(self) -> None:
            self.trial = trial
            self.report_metric = report_metric
            self.prune = prune
            self.pruned = False
            self.last_value: Optional[float] = None
            self.n_reports = 0

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not metrics:
                return control
            value = metrics.get(self.report_metric)
            if value is None:
                for key, candidate in metrics.items():
                    if key.startswith("eval_") and isinstance(candidate, (int, float)):
                        value = candidate
                        break
            if value is None:
                return control
            self.last_value = float(value)
            self.n_reports += 1
            self.trial.report(float(value), step=int(state.global_step))
            if self.prune and self.trial.should_prune():
                self.pruned = True
                control.should_training_stop = True
            return control

    _ = optuna  # ensure optuna import is required at call time
    return OptunaPruningCallback()
