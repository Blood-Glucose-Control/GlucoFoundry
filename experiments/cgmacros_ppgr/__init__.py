# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""CGMacros post-meal glucose response (PPGR) with frozen CGM representations experiment recipe.

How well do frozen CGM foundation-model embeddings of the 24 h before a meal predict the 2 h post-meal glucose response (GlucoFM paper, Table 4)?

Self-contained recipe: episode construction (``episodes``), evaluation
(``eval``), and result summarization (``summarize``). Registered with the core
via :mod:`register`, so the ``experiments/cgmacros_ppgr/`` directory can
be dropped cleanly without touching the toolkit core.
"""
