# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Shared drawing helpers for the GlucoFoundry workflow diagrams.

Small matplotlib primitives (rounded boxes, arrows, anchor points) and the
shared colour palette, so each ``render_*.py`` script stays short and the
diagrams keep a consistent house style. Only dependency is matplotlib.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Shared palette (matches render_glucofoundry_workflow.py).
IO = dict(fc="#CFF3EE", ec="#2C7A7B", tc="#1D4044")  # data in/out
CORE = dict(fc="#D6E4FF", ec="#2B6CB0", tc="#1A365D")  # pipeline / core steps
MODEL = dict(fc="#D7F5DD", ec="#2F855A", tc="#1C4532")  # models / workers
STORE = dict(fc="#FFF4D6", ec="#D69E2E", tc="#744210")  # storage / outputs
NOTE = dict(fc="#F7FAFC", ec="#A0AEC0", tc="#2D3748")  # side notes


def box(ax, cx, cy, w, h, text, style, *, dashed=False, fontsize=11, bold=True):
    """Draw a centered rounded box and return ``(cx, cy, w, h)``."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=1.8,
        facecolor=style["fc"],
        edgecolor=style["ec"],
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        cx,
        cy,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=style["tc"],
        fontweight="bold" if bold else "normal",
        zorder=3,
    )
    return (cx, cy, w, h)


def arrow(ax, p0, p1, *, color="#4A5568", dashed=False, rad=0.0, lw=2.0, label=None):
    """Draw an arrow from point ``p0`` to ``p1`` with an optional italic label."""
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=lw,
            color=color,
            linestyle="--" if dashed else "-",
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
    )
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(
            mx + 0.1,
            my + 0.22,
            label,
            ha="center",
            va="bottom",
            fontsize=9.5,
            style="italic",
            color=color,
            zorder=3,
        )


def top(b):
    cx, cy, w, h = b
    return (cx, cy + h / 2)


def bottom(b):
    cx, cy, w, h = b
    return (cx, cy - h / 2)


def left(b):
    cx, cy, w, h = b
    return (cx - w / 2, cy)


def right(b):
    cx, cy, w, h = b
    return (cx + w / 2, cy)


def save(fig, out_dir, stem):
    """Write ``<stem>.png`` and ``<stem>.svg`` next to the diagrams."""
    png = out_dir / f"{stem}.png"
    svg = out_dir / f"{stem}.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    print(f"wrote {png}")
    print(f"wrote {svg}")
