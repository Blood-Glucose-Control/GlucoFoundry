# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Render the `glucofoundry new-experiment` scaffolder workflow diagram.

Interview (or --from-file) -> validate/derive -> render templates/experiment/
into a self-contained experiments/<name>/ that the registry auto-discovers.

    python agent_notes/diagrams/render_scaffolder_workflow.py

Only dependency is matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from _diagram_common import (
    CORE,
    IO,
    MODEL,
    STORE,
    arrow,
    bottom,
    box,
    left,
    right,
    save,
    top,
)


def main() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 10.0))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(-0.4, 15)
    ax.axis("off")

    ax.text(
        7.25,
        14.4,
        "glucofoundry new-experiment",
        ha="center",
        fontsize=19,
        fontweight="bold",
        color="#1A365D",
    )
    ax.text(
        7.25,
        13.8,
        "scaffold a self-contained experiment from the template",
        ha="center",
        fontsize=12,
        style="italic",
        color="#4A5568",
    )

    # Left column: inputs -> processing -> render
    u = box(
        ax,
        4.2,
        12.4,
        4.6,
        1.0,
        "developer runs\nglucofoundry new-experiment",
        IO,
        fontsize=10,
    )
    q = box(
        ax,
        4.2,
        10.4,
        4.6,
        1.2,
        "Interview\n(templates/experiment.yaml)\n+ registry lookup: datasets, models",
        CORE,
        fontsize=9.5,
    )
    ff = box(
        ax,
        11.6,
        12.4,
        4.4,
        1.0,
        "--from-file\nanswers.yaml (non-interactive)",
        IO,
        fontsize=9.5,
    )
    v = box(
        ax,
        4.2,
        8.2,
        4.6,
        1.2,
        "Validate / coerce answers\nderive summarizer_class,\nexperiment_type",
        CORE,
        fontsize=9.5,
    )
    r = box(
        ax,
        4.2,
        6.0,
        4.6,
        1.1,
        "Render templates/experiment/\nreplace {{tokens}}, strip .tmpl",
        CORE,
        fontsize=10,
    )
    t = box(
        ax,
        11.6,
        6.0,
        4.0,
        1.1,
        "templates/experiment/\nstubs + baselines",
        MODEL,
        fontsize=10,
    )

    arrow(ax, bottom(u), top(q))
    arrow(
        ax,
        left(ff),
        (right(q)[0], right(q)[1] - 0.1),
        dashed=True,
        rad=-0.1,
        label="non-interactive",
    )
    arrow(ax, bottom(q), top(v))
    arrow(ax, bottom(v), top(r))
    arrow(ax, left(t), right(r))

    # Output: experiments/<name>/ tree
    e = box(ax, 7.25, 3.9, 6.6, 1.0, "experiments/<name>/", STORE, fontsize=12)
    e1 = box(ax, 2.5, 1.9, 4.0, 1.0, "register.py\n-> ExperimentSpec", IO, fontsize=9.5)
    e2 = box(
        ax, 7.25, 1.9, 4.2, 1.0, "episodes / eval /\nsummarize stubs", IO, fontsize=9.5
    )
    e3 = box(
        ax,
        11.9,
        1.9,
        4.2,
        1.0,
        "configs/ (data,\nmodels/<selected>, sweeps)",
        IO,
        fontsize=9,
    )

    arrow(ax, bottom(r), (top(e)[0] - 1.5, top(e)[1]), rad=-0.1)
    arrow(ax, bottom(e), top(e1), rad=0.1)
    arrow(ax, bottom(e), top(e2))
    arrow(ax, bottom(e), top(e3), rad=-0.1)

    disc = box(
        ax,
        2.5,
        0.3,
        4.0,
        0.72,
        "auto-discovered by\nsrc/experiments/registry.py",
        CORE,
        fontsize=8.5,
        bold=False,
    )
    arrow(ax, bottom(e1), top(disc), color="#2B6CB0")

    out = Path(__file__).resolve().parent
    save(fig, out, "scaffolder_workflow")


if __name__ == "__main__":
    main()
