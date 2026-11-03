# Copyright (c) 2025 Blood-Glucose-Control
# Licensed under Custom Research License (see LICENSE file)

"""Top-level experiment recipes.

Each subdirectory is a self-contained, droppable experiment recipe that imports
the pure ``src/`` core and self-registers via its ``register.py`` module (see
``docs/design/toolkit-charter.md``). This directory is importable in-repo
(``pythonpath = ["."]``) but excluded from the installed wheel
(``packages.find = ["src*"]``).
"""
