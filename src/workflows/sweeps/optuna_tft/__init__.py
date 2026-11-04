# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Temporal Fusion Transformer (TFT) x Optuna sweep on MetaboNet.

Reuses the Chronos-2 sweep data/metrics/feature pipeline; swaps the model to
AutoGluon TemporalFusionTransformerModel with its own search space.
"""
