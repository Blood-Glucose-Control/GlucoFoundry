# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Top-level experiment recipes.

Each subdirectory is a self-contained, droppable experiment recipe that imports
the pure ``src/`` core and self-registers via its ``register.py`` module (see
``docs/architecture/toolkit-architecture.md``). This directory is importable in-repo
(``pythonpath = ["."]``) but excluded from the installed wheel
(``packages.find = ["src*"]``).
"""
