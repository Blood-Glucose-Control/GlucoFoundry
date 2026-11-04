# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(file_path: str | Path) -> Any:
    """Load and parse a YAML configuration file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
