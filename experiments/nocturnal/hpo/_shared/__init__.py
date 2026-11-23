# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Shared building blocks for the nocturnal HPO sweeps.

- :mod:`.data`      - loads train/holdout splits and exposes them to the engine.
- :mod:`.objective` - the model-agnostic per-trial objective (train + evaluate
  on the reported core metrics, optimizing WQL).
"""
