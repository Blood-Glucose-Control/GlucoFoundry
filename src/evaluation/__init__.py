"""Evaluation modules (generic core).

Experiment-specific evaluation (e.g. nocturnal) lives in its own recipe under
``experiments/<name>/`` and is resolved via the experiment registry.
"""

from .episode_builders import build_midnight_episodes

__all__ = [
    "build_midnight_episodes",
]
