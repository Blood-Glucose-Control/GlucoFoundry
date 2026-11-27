# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Run the CGMacros PPGR task with a frozen CGM-JEPA representation.

Stages follow the GlucoFM paper's Table 4, each adding inputs to the head:

* ``frozen``: the mean-pooled CGM-JEPA embedding of the 24 h before the meal
* ``recent_cgm``: + the last hour of glucose before the meal (the last 12
  five-minute slots of the input, mg/dL; for Libre these are ~4 readings
  filled linearly). Also reports a head on the last hour alone, to show what
  the embedding adds on top.

Dexcom and Libre are trained and scored separately, then averaged.

Usage::

    python -m experiments.cgmacros_ppgr.run
    python -m experiments.cgmacros_ppgr.run --checkpoint x_cgm_jepa --participants mets34
    python -m experiments.cgmacros_ppgr.run --stage recent_cgm

Results go to ``results/<experiment_type>/288ctx_24fh/<checkpoint>/<run>/``;
summarize them with :class:`~experiments.cgmacros_ppgr.summarize.CgmacrosPpgrSummarizer`.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.data.cache_manager import get_cache_manager
from src.data.diabetes_datasets.cgmacros.data_cleaner import (
    RAW_COLS,
    RAW_SUBDIR,
    load_raw_cgmacros_data,
)
from src.experiments.registry import get_experiment_paths
from src.models.cgm_jepa.encoder import CgmJepaEncoder

from .episodes import (
    CONTEXT_STEPS,
    MealEvents,
    build_episodes,
    load_events,
    save_events,
)
from .eval import METRICS, evaluate
from .summarize import EXPERIMENT_TYPE

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "cgmacros_ppgr"
CTX_FH = "288ctx_24fh"
STAGES = ("frozen", "recent_cgm")
RECENT_STEPS = 12  # last hour of the 5-min input

# GlucoFM paper, Table 4, mean of Dexcom and Libre, in METRICS order
# (trajectory, positive iAUC, peak rise, peak time). Taken from a summary of
# the paper; check against the PDF.
PAPER = {
    "frozen": {
        "cgm_jepa": (26.50, 36.89, 31.36, 32.27),
        "x_cgm_jepa": (26.19, 35.78, 31.24, 32.19),
    },
    "recent_cgm": {
        "cgm_jepa": (24.82, 33.73, 29.15, 32.66),
        "x_cgm_jepa": (25.14, 34.09, 29.96, 32.85),
    },
}


def load_meal_events(
    participants: str, rebuild: bool, cache_dir: Path
) -> dict[str, MealEvents]:
    """Build (or load cached) meal events for all participants or the METs 34."""
    path = cache_dir / f"events_{participants}.npz"
    if path.exists() and not rebuild:
        return load_events(path)
    raw_dir = (
        get_cache_manager().get_absolute_path_by_type("cgmacros", "raw") / RAW_SUBDIR
    )
    raw = load_raw_cgmacros_data(raw_dir)
    if participants == "mets34":
        raw = {pid: df for pid, df in raw.items() if RAW_COLS["mets"] in df.columns}
    events = build_episodes(raw)
    save_events(events, path)
    return events


@torch.no_grad()
def embed(contexts: np.ndarray, checkpoint: str) -> np.ndarray:
    """Mean-pooled frozen CGM-JEPA embedding, (N, 288) mg/dL -> (N, 96)."""
    encoder = CgmJepaEncoder.from_pretrained(subfolder=checkpoint).eval()
    x = torch.tensor(contexts, dtype=torch.float32)
    patches = x.reshape(len(x), CONTEXT_STEPS // encoder.patch_len, encoder.patch_len)
    return encoder(patches).mean(dim=1).numpy()


def head_inputs(ev: MealEvents, checkpoint: str, stage: str) -> dict[str, np.ndarray]:
    """Feature sets to train a head on: ``model`` plus any comparison heads."""
    emb = embed(ev.contexts, checkpoint)
    if stage == "frozen":
        return {"model": emb}
    recent = ev.contexts[:, -RECENT_STEPS:]
    return {"model": np.hstack([emb, recent]), "last_hour_only": recent}


def summarize_seeds(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        m: {
            "mean": float(np.mean([r[m] for r in per_seed])),
            "sd": float(np.std([r[m] for r in per_seed])),
        }
        for m in METRICS
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CGMacros PPGR with a frozen CGM-JEPA representation"
    )
    parser.add_argument(
        "--checkpoint", choices=["cgm_jepa", "x_cgm_jepa"], default="cgm_jepa"
    )
    parser.add_argument("--participants", choices=["all", "mets34"], default="all")
    parser.add_argument("--stage", choices=STAGES, default="frozen")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rebuild-events", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    paths = get_experiment_paths(EXPERIMENT_NAME).ensure()
    events = load_meal_events(
        args.participants, args.rebuild_events, paths.results / "_cache"
    )
    n_meals = len(events["dexcom"].patient_ids)
    n_people = len(set(events["dexcom"].patient_ids))
    print(f"\nMeals: {n_meals} from {n_people} participants (paper: 874 from 34)")

    per_seed: dict[str, dict[str, list]] = {}
    for sensor, ev in events.items():
        inputs = head_inputs(ev, args.checkpoint, args.stage)
        per_seed[sensor] = evaluate(ev, inputs, args.seeds)

    # Two-sensor mean: average Dexcom and Libre per seed, then over seeds
    per_seed["mean"] = {
        method: [
            {m: (d[m] + lib[m]) / 2 for m in METRICS}
            for d, lib in zip(per_seed["dexcom"][method], per_seed["libre"][method])
        ]
        for method in per_seed["dexcom"]
    }
    summary = {
        sensor: {method: summarize_seeds(runs) for method, runs in by_method.items()}
        for sensor, by_method in per_seed.items()
    }

    names = {
        "trajectory": "Trajectory",
        "iauc_pos": "Pos. iAUC",
        "peak_rise": "Peak rise",
        "peak_time": "Peak time",
    }
    print(
        f"\nMAE, stage={args.stage}, checkpoint={args.checkpoint}, "
        f"participants={args.participants}, seeds={args.seeds}"
    )
    print(f"{'':24}" + "".join(f"{names[m]:>16}" for m in METRICS))
    for sensor in ("dexcom", "libre", "mean"):
        for method in per_seed[sensor]:
            s = summary[sensor][method]
            cells = "".join(
                f"{s[m]['mean']:>9.2f} ± {s[m]['sd']:<4.2f}" for m in METRICS
            )
            print(f"{sensor + ' / ' + method:24}{cells}")
    paper = dict(zip(METRICS, PAPER[args.stage][args.checkpoint]))
    print(f"{'paper (mean)':24}" + "".join(f"{paper[m]:>16.2f}" for m in METRICS))
    print("Units: mg/dL, mg/dL*h, mg/dL, min")

    run_name = (
        f"{datetime.now():%Y-%m-%d_%H%M%S}_cgmacros_{args.stage}_{args.participants}"
    )
    run_dir = paths.results / EXPERIMENT_TYPE / CTX_FH / args.checkpoint / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "results.json"
    out.write_text(
        json.dumps(
            {
                "args": vars(args),
                "meals": n_meals,
                "participants": n_people,
                "summary": summary,
                "paper": paper,
            },
            indent=2,
        )
    )
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
