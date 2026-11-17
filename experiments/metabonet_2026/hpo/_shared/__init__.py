# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Shared MetaboNet-2026 sweep building blocks used by every model family.

- :mod:`.data`     - ``MetabonetData`` panels/val-window assembly + ``build_data``
- :mod:`.features` - MetaboNet feature schema + covariate candidate columns
- :mod:`.kinetics` - insulin/carb kinetics feature engineering
- :mod:`.metrics`  - MetaboNet challenge metrics (RMSE/MARD/DTS, ``evaluate_horizons``)
"""
