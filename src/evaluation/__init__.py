"""Evaluation modules (generic core).

Experiment-specific evaluation (e.g. nocturnal) lives in its own recipe under
``experiments/<name>/`` and is resolved via the experiment registry.
"""

from .episode_builders import get_holdout_patients, select_episodes_stratified

__all__ = [
    "get_holdout_patients",
    "select_episodes_stratified",
]
