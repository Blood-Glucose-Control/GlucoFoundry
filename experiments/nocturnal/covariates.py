# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Covariate bucketing and model-capability lookup for the nocturnal recipe.

Extracted from the (removed) grand-summary table builder: the reusable pieces the
core evaluation/summarization path depends on — the per-model capability registry
(:data:`MODEL_PROPERTIES`), covariate-bucket classification
(:func:`bucket_from_covariates`), and run-config covariate reading
(:func:`read_covariate_cols`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# Buckets for the covariate configuration of a run.
COV_BUCKETS: tuple[str, ...] = ("zero_shot", "bg_only", "iob", "iob_cob")

# Architecture-level capabilities per model family.
# Keys match the `model` column emitted by NocturnalSummarizer (multi-variant
# models keep their `parent/variant` form, e.g. `statistical/AutoARIMA`).
#
# Booleans:
#   zero_shot_capable       -- pretrained weights usable with no fine-tune?
#   fine_tunable            -- supports per-dataset fine-tune in our pipeline?
#   supports_past_covariates-- accepts past covariates (IOB / COB / IA)?
#   univariate_only         -- forced to single channel?
#   probabilistic           -- emits quantiles / sample paths?
MODEL_PROPERTIES: dict[str, dict[str, Any]] = {
    # ---- Naive baselines ----
    # Naive baselines are deterministic; we evaluate them through the same
    # pipeline as fine-tuned models (mode=finetuned) so we mark them
    # fine_tunable rather than zero_shot_capable to match how the runs are
    # recorded.  Conceptually they have no learnable parameters either way.
    "naive_baseline/Naive": {
        "class": "naive",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": False,
        "architecture_summary": (
            "Last-observation-carried-forward. Deterministic; no learnable parameters."
        ),
    },
    "naive_baseline/Average": {
        "class": "naive",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": False,
        "architecture_summary": (
            "Mean of the context window. Deterministic; no learnable parameters."
        ),
    },
    # ---- Statistical ----
    "statistical/AutoARIMA": {
        "class": "statistical",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": True,
        "architecture_summary": (
            "Box-Jenkins ARIMA with automated (p,d,q) order selection per series; "
            "Gaussian predictive distribution."
        ),
    },
    "statistical/Theta": {
        "class": "statistical",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": False,
        "architecture_summary": (
            "Theta decomposition: drift line + simple exponential smoothing of "
            "deseasonalised series. Point forecasts only."
        ),
    },
    "statistical/NPTS": {
        "class": "statistical",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": True,
        "architecture_summary": (
            "Non-Parametric Time Series: empirical resampling of historical values "
            "with kernel-weighted recency."
        ),
    },
    # ---- Deep learning (trained from scratch / on our data) ----
    "deepar": {
        "class": "deep_learning",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": False,
        "probabilistic": True,
        "architecture_summary": (
            "Autoregressive LSTM emitting parametric (Student-t/Gaussian) "
            "likelihood per step. AutoGluon only exposes known (future-available) "
            "covariates for DeepAR; past-only covariates like IOB/COB cannot be "
            "used without data leakage."
        ),
    },
    "tft": {
        "class": "deep_learning",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": True,
        "architecture_summary": (
            "Temporal Fusion Transformer: variable-selection networks + LSTM "
            "encoder/decoder + multi-head interpretable attention + quantile head."
        ),
    },
    "patchtst": {
        "class": "deep_learning",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": False,
        "architecture_summary": (
            "Patch-tokenised transformer encoder, channel-independent. "
            "Point/quantile forecasts; no native covariate support."
        ),
    },
    "tide": {
        "class": "deep_learning",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": False,
        "probabilistic": False,
        "architecture_summary": (
            "Time-series Dense Encoder: pure MLP encoder-decoder with residual "
            "connections; no attention. Point forecasts. AutoGluon only exposes "
            "known (future-available) covariates for TiDE; past-only covariates "
            "like IOB/COB cannot be used without data leakage."
        ),
    },
    "tsmixer": {
        "class": "deep_learning",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": True,
        "architecture_summary": (
            "Time-Series Mixer (all-MLP): stacked temporal/feature mixing blocks "
            "without attention. Darts-backed implementation supports past "
            "covariate channels (e.g., IOB/COB) and quantile forecasts."
        ),
    },
    "timegrad": {
        "class": "deep_learning",
        "zero_shot_capable": False,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": True,
        "architecture_summary": (
            "Autoregressive denoising-diffusion RNN: LSTM hidden state conditions "
            "a per-step DDPM that samples next observation. Uses the pts library "
            "directly (not AutoGluon); only BG context is consumed — no covariate "
            "inputs are wired in."
        ),
    },
    # ---- Foundation (pre-trained) ----
    "chronos2": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": True,
        "architecture_summary": (
            "T5 encoder-decoder over quantised value tokens (Chronos-2). "
            "Native past + static covariates; quantile sampling."
        ),
    },
    "moirai": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": True,
        "architecture_summary": (
            "Masked encoder transformer with patch embeddings and multi-patch-size "
            "attention; any-variate forecasting."
        ),
    },
    "toto": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": True,
        "architecture_summary": (
            "Datadog Toto: decoder-only transformer foundation model with "
            "proprietary tokeniser, trained on observability time-series."
        ),
    },
    "timesfm": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": True,
        "architecture_summary": (
            "Google TimesFM: decoder-only transformer with input/output patching; "
            "univariate; quantile head."
        ),
    },
    "ttm": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": False,
        "architecture_summary": (
            "TinyTimeMixer (IBM): mixer-style channel/patch MLPs (no attention); "
            "very small (~1M params). Point forecasts."
        ),
    },
    "sundial": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": False,
        "supports_past_covariates": False,
        "univariate_only": True,
        "probabilistic": True,
        "architecture_summary": (
            "THUML Sundial: generative diffusion-based transformer foundation model "
            "(ICML 2025). Draws sample paths via autoregressive generation; "
            "quantiles derived from samples. Used zero-shot only in this study."
        ),
    },
    "moment": {
        "class": "foundation",
        "zero_shot_capable": True,
        "fine_tunable": True,
        "supports_past_covariates": True,
        "univariate_only": False,
        "probabilistic": False,
        "architecture_summary": (
            "MOMENT (CMU): masked-reconstruction transformer foundation model "
            "with patch embeddings; channel-independent variant supports past "
            "covariates via concatenation. Point forecasts."
        ),
    },
}


def model_supports_past_covariates(model_name: str) -> bool:
    """Return True if *model_name* can consume past covariate columns.

    Looks up ``MODEL_PROPERTIES`` by exact key first, then by the base model
    name (stripping any ``/variant`` suffix).  Returns ``False`` for unknown
    models so that unrecognised names are treated conservatively.
    """
    if model_name in MODEL_PROPERTIES:
        return MODEL_PROPERTIES[model_name]["supports_past_covariates"]
    base = model_name.split("/")[0]
    if base in MODEL_PROPERTIES:
        return MODEL_PROPERTIES[base]["supports_past_covariates"]
    return False


def read_covariate_cols(run_path: str | Path) -> list[str] | None:
    """Read `cli_args.covariate_cols` from a run's experiment_config.json."""
    cfg_path = Path(run_path) / "experiment_config.json"
    if not cfg_path.exists():
        return None
    try:
        with cfg_path.open() as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not read %s: %s", cfg_path, exc)
        return None
    cli = cfg.get("cli_args", {}) or {}
    cols = cli.get("covariate_cols")
    if cols is None:
        # Fall back to model_config.covariate_cols (populated when covariates
        # are specified in the model YAML rather than as CLI args).
        model_cfg = cfg.get("model_config", {}) or {}
        cols = model_cfg.get("covariate_cols")
    if cols is None:
        return []
    if isinstance(cols, list):
        return [str(c) for c in cols]
    return []


_CARB_COVARIATES: frozenset[str] = frozenset({"cob", "carb_availability", "carbs"})
# Insulin-related (beyond plain IOB) covariates that count as IOB-family signal.
_INSULIN_COVARIATES: frozenset[str] = frozenset({"iob", "ia", "insulin_availability"})


def bucket_from_covariates(mode: str, cov_cols: Iterable[str] | None) -> str:
    """Map (mode, covariate_cols) to one of COV_BUCKETS.

    ``mode`` should be the normalised string produced by the summarizer
    (hyphens stripped, lower-cased), e.g. ``"zeroshot"`` or ``"finetuned"``.

    Buckets:
        zero_shot -- model run in zero-shot mode
        iob_cob   -- both insulin (IOB / insulin_availability) and carb signals present
        iob       -- insulin signal only: any of iob, ia, insulin_availability
        bg_only   -- no non-BG covariates (cob-only configurations are not valid
                     and are treated as bg_only with a warning)
    """
    if mode == "zeroshot":
        return "zero_shot"
    cset = {c.lower() for c in (cov_cols or [])}
    # Drop time-feature names that aren't clinical BG covariates
    cset -= {"hour_sin", "hour_cos", "minute_sin", "minute_cos", "day_sin", "day_cos"}
    if not cset:
        return "bg_only"
    has_carb = bool(cset & _CARB_COVARIATES)
    has_insulin = bool(cset & _INSULIN_COVARIATES)
    if has_carb and has_insulin:
        return "iob_cob"
    if has_insulin:
        return "iob"
    if has_carb:
        # COB-only is not a valid configuration; flag and fall back to bg_only
        log.warning(
            "Carb-only covariate set %s — no insulin covariates present. "
            "COB-only is not a valid bucket; bucketing as bg_only.",
            sorted(cset),
        )
        return "bg_only"
    # Unknown covariate set: treat as BG-only and warn
    log.warning("Unrecognised covariate set %s — bucketing as bg_only", sorted(cset))
    return "bg_only"
