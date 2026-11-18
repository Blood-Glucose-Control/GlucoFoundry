# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""MetaboNet Glucose Prediction Challenge (2026) experiment recipe.

Self-contained experiment code for the MetaboNet 2026 submission: the
hyperparameter-optimization sweep tree (:mod:`experiments.metabonet_2026.hpo`)
and its submission tooling. It imports the pure ``src/`` core (notably the
reusable HPO engine at :mod:`src.workflows.hpo.engine`) and is importable
in-repo (``pythonpath = ["."]``) but excluded from the installed wheel
(``packages.find = ["src*"]``).
"""
