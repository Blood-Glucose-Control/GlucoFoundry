# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Result summarizer for the cgmacros_ppgr experiment."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.experiments.base.experiment import ExperimentSummarizer
from src.experiments.registry import get_experiment_paths

log = logging.getLogger(__name__)

#: Run-artifact subdirectory this experiment writes under
#: (``<results>/<experiment_type>/...``).
EXPERIMENT_TYPE = "cgmacros_ppgr_representation_learning"


class CgmacrosPpgrSummarizer(ExperimentSummarizer):
    """Summarize runs under this experiment's ``results/<experiment_type>/``.

    One row per run: the two-sensor mean of the model head's endpoint MAEs
    (``mae`` is the trajectory MAE), plus the baselines' trajectory MAE.
    """

    def __init__(self, experiments_root: str | Path | None = None) -> None:
        if experiments_root is None:
            experiments_root = get_experiment_paths("cgmacros_ppgr").results
        super().__init__(experiments_root, EXPERIMENT_TYPE)

    def _parse_run_dir(
        self,
        ctx_fh: str,
        model: str,
        run_dir: Path,
    ) -> dict[str, Any] | None:
        path = run_dir / "results.json"
        try:
            data = json.loads(path.read_text())
            mean = data["summary"]["mean"]
            head = mean["model"]
        except (OSError, ValueError, KeyError) as exc:
            log.debug("Skipping %s: %s", run_dir, exc)
            return None
        args = data.get("args", {})
        row: dict[str, Any] = {
            "run": run_dir.name,
            "ctx_fh": ctx_fh,
            "model": model,
            "dataset": "cgmacros",
            "stage": args.get("stage"),
            "participants": args.get("participants"),
            "seeds": args.get("seeds"),
            "meals": data.get("meals"),
            "mae": head["trajectory"]["mean"],
            "mae_sd": head["trajectory"]["sd"],
        }
        for metric in ("iauc_pos", "peak_rise", "peak_time"):
            row[metric] = head[metric]["mean"]
        for baseline in ("zero", "mean_curve", "last_hour_only"):
            if baseline in mean:
                row[f"{baseline}_mae"] = mean[baseline]["trajectory"]["mean"]
        return row
