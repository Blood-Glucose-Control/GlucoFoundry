# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Render the Optuna sweep workflow diagram.

How src/workflows/sweeps/optuna_<family>/ works: a shared study on RDB storage,
cooperative GPU workers each running study.optimize(objective), and result export.

    python agent_notes/diagrams/render_optuna_sweep_workflow.py

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
    fig, ax = plt.subplots(figsize=(15.0, 9.5))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.axis("off")

    ax.text(
        7.5,
        14.4,
        "Optuna sweep workflow",
        ha="center",
        fontsize=19,
        fontweight="bold",
        color="#1A365D",
    )
    ax.text(
        7.5,
        13.8,
        "cooperative, shared-study hyperparameter optimization",
        ha="center",
        fontsize=12,
        style="italic",
        color="#4A5568",
    )

    # Setup column (left)
    cfg = box(
        ax,
        3.0,
        12.2,
        4.4,
        1.3,
        "study config (YAML)\nstudy_name, storage, search space,\nsampler, pruner, direction",
        IO,
        fontsize=9.5,
    )
    create = box(
        ax,
        3.0,
        10.1,
        4.4,
        1.1,
        "study.py create\noptuna.create_study(load_if_exists)",
        CORE,
        fontsize=10,
    )
    store = box(
        ax,
        3.0,
        7.7,
        4.6,
        1.6,
        "Shared Optuna study\nRDB storage + heartbeat\n(TPE sampler, median pruner)",
        STORE,
        fontsize=10,
    )

    # Launcher + workers (right)
    launch = box(
        ax,
        11.0,
        12.2,
        4.0,
        1.1,
        "launch_optuna_tmux.sh\nGPUS, TRIALS_PER_GPU",
        CORE,
        fontsize=10,
    )
    w1 = box(
        ax, 9.3, 10.1, 2.9, 1.0, "worker (GPU 0)\nstudy.py run", MODEL, fontsize=10
    )
    w2 = box(
        ax, 12.7, 10.1, 2.9, 1.0, "worker (GPU 1)\nstudy.py run", MODEL, fontsize=10
    )

    # Per-trial objective loop
    obj = box(ax, 11.0, 7.9, 4.8, 1.0, "objective(trial)", CORE, fontsize=11)
    ss = box(ax, 11.0, 6.4, 4.8, 0.95, "search_space.suggest(trial)", CORE, fontsize=10)
    tr = box(ax, 11.0, 4.9, 4.8, 0.95, "train model (ModelAdapter)", CORE, fontsize=10)
    rp = box(ax, 11.0, 3.4, 4.8, 0.95, "report val loss -> prune?", CORE, fontsize=10)

    # Export / results (bottom-left)
    exp = box(
        ax,
        3.0,
        4.0,
        4.6,
        1.2,
        "export_results / build_submission\nbest config, leaderboard, ensemble",
        CORE,
        fontsize=9.5,
    )
    res = box(ax, 3.0, 1.7, 4.0, 1.0, "Reproducible results", IO, fontsize=11)

    arrow(ax, bottom(cfg), top(create))
    arrow(ax, bottom(create), top(store))
    arrow(ax, bottom(launch), (top(w1)[0], top(w1)[1]), rad=0.15)
    arrow(ax, bottom(launch), (top(w2)[0], top(w2)[1]), rad=-0.15)
    arrow(ax, bottom(w1), (top(obj)[0] - 0.6, top(obj)[1]), rad=0.1)
    arrow(ax, bottom(w2), (top(obj)[0] + 0.6, top(obj)[1]), rad=-0.1)
    ax.text(
        11.0,
        9.0,
        "study.optimize(objective, n_trials)",
        ha="center",
        fontsize=9.5,
        style="italic",
        color="#2F855A",
    )
    arrow(ax, bottom(obj), top(ss))
    arrow(ax, bottom(ss), top(tr))
    arrow(ax, bottom(tr), top(rp))

    # trials written back to the shared store
    arrow(
        ax,
        left(obj),
        (right(store)[0], right(store)[1] + 0.2),
        color="#D69E2E",
        rad=0.18,
        label="write trial",
    )
    arrow(
        ax,
        left(rp),
        (right(store)[0] + 0.1, right(store)[1] - 0.5),
        color="#D69E2E",
        dashed=True,
        rad=0.3,
        label="TrialPruned",
    )
    # workers connect to the shared store
    arrow(ax, bottom(store), top(exp))
    arrow(ax, bottom(exp), top(res))

    out = Path(__file__).resolve().parent
    save(fig, out, "optuna_sweep_workflow")


if __name__ == "__main__":
    main()
