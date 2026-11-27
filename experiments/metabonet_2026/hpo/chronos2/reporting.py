# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

# pyright: reportMissingImports=false
"""Reporting/pruning-aware Chronos-2 model subclass.

``Chronos2Model._fine_tune`` builds its HuggingFace ``Trainer`` callback list
inline (there is no ``_get_callbacks`` hook like ``DeepARModel`` has), but it
loads its pipeline through the overridable ``load_model_pipeline()`` and then
calls ``self._model_pipeline.fit(callbacks=...)``. :class:`ReportingChronos2Model`
hooks that method to append the Optuna callbacks bound via
:class:`~experiments.metabonet_2026.hpo.chronos2.pruning.active_report_callbacks`
— a scoped, auto-registered alternative to globally monkey-patching
``Chronos2Pipeline.fit``.

Pass the class itself as a ``TimeSeriesPredictor.fit`` hyperparameters key
(``{ReportingChronos2Model: params}``). Imported lazily (it imports AutoGluon at
module load) only when reporting/pruning is enabled.
"""

from __future__ import annotations

from autogluon.timeseries.models.chronos.chronos2 import Chronos2Model

from .pruning import append_active_callbacks


class ReportingChronos2Model(Chronos2Model):
    """Chronos2Model that forwards the active Optuna callbacks to its Trainer."""

    def load_model_pipeline(self) -> None:
        super().load_model_pipeline()
        append_active_callbacks(self._model_pipeline)
