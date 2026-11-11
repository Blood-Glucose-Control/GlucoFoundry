# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Render the single-model training/eval workflow diagram.

The "one run" that Optuna wraps per trial: config -> orchestrator -> pipeline
(load -> build model -> train -> evaluate) -> experiment-owned results.

    python agent_notes/diagrams/render_single_model_training_workflow.py

Only dependency is matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from _diagram_common import (
    CORE,
    IO,
    MODEL,
    NOTE,
    arrow,
    bottom,
    box,
    left,
    right,
    save,
    top,
)


def main() -> None:
    fig, ax = plt.subplots(figsize=(17.5, 6.0))
    ax.set_xlim(0, 19.5)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(
        9.75,
        6.6,
        "Single-model training / evaluation",
        ha="center",
        fontsize=18,
        fontweight="bold",
        color="#1A365D",
    )
    ax.text(
        9.75,
        6.1,
        "the inner loop Optuna runs once per trial",
        ha="center",
        fontsize=12,
        style="italic",
        color="#4A5568",
    )

    y = 4.2
    mc = box(
        ax,
        1.9,
        5.1,
        2.9,
        1.0,
        "model config\nconfigs/models/<family>/*.yaml",
        IO,
        fontsize=9.5,
    )
    dc = box(
        ax,
        1.9,
        3.3,
        2.9,
        1.0,
        "data / holdout config\n+ --experiment <name>",
        IO,
        fontsize=9.5,
    )
    o = box(
        ax,
        5.6,
        y,
        3.0,
        1.15,
        "orchestrator CLI\n(forecasting pipeline)",
        CORE,
        fontsize=10,
    )
    ld = box(ax, 9.0, y, 2.4, 1.15, "load +\npreprocess data", CORE, fontsize=10)
    m = box(ax, 11.9, y, 2.4, 1.15, "build model\n(ModelAdapter)", MODEL, fontsize=10)
    tr = box(ax, 14.6, y, 2.1, 1.15, "train /\nfine-tune", CORE, fontsize=10)
    ev = box(ax, 17.1, y, 2.1, 1.15, "evaluate\n-> metrics", CORE, fontsize=10)
    r = box(
        ax,
        17.1,
        1.7,
        3.4,
        1.1,
        "experiments/<name>/results/\nresults_summary + run_manifest.json",
        IO,
        fontsize=9,
    )

    arrow(ax, right(mc), (left(o)[0], left(o)[1] + 0.25))
    arrow(ax, right(dc), (left(o)[0], left(o)[1] - 0.25))
    arrow(ax, right(o), left(ld))
    arrow(ax, right(ld), left(m))
    arrow(ax, right(m), left(tr))
    arrow(ax, right(tr), left(ev))
    arrow(ax, bottom(ev), top(r))

    note = box(
        ax,
        6.5,
        1.9,
        6.6,
        1.2,
        "Optuna's objective() runs exactly this inner loop per trial,\n"
        "with hyperparameters sampled from the search space.",
        NOTE,
        dashed=True,
        fontsize=10,
        bold=False,
    )
    arrow(
        ax,
        top(note),
        (bottom(o)[0], bottom(o)[1]),
        color="#A0AEC0",
        dashed=True,
        rad=-0.15,
    )

    out = Path(__file__).resolve().parent
    save(fig, out, "single_model_training_workflow")


if __name__ == "__main__":
    main()
