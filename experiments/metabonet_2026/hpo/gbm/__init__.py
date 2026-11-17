# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""GBM (gradient-boosted tree) x Optuna sweep on MetaboNet.

Reuses the Chronos-2 sweep data/metrics/feature pipeline; swaps the model to
AutoGluon's tabular reductions (RecursiveTabular / DirectTabular) backed by
LightGBM / CatBoost / XGBoost, with its own known-future covariate search
space. CPU-only, so it can run concurrently with GPU-bound sweeps.
"""
