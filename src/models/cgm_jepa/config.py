# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the CGM-JEPA encoder + forecast-head model."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, List, Optional

from ..base import ModelConfig, TrainingBackend

MGDL_PER_MMOL = 18.018


class CgmJepaConfig(ModelConfig):
    """``ModelConfig`` plus the CGM-JEPA encoder / head knobs.

    CGM-JEPA is a *representation* model (24 h window -> hourly 96-d tokens);
    it has no forecasting head upstream. This config describes the encoder
    that is loaded/fine-tuned and the small head trained on top of it.

    Attributes:
        pretrained_subfolder: HF subfolder (``"cgm_jepa"`` / ``"x_cgm_jepa"``) or
            ``None`` for a randomly initialised encoder (ablation / offline tests).
        pretrained_repo_id: HF repo holding the published weights (defaults to
            ``model_path`` when that is set).
        pretrained_local_dir: Optional local mirror of the HF repo layout.
        freeze_encoder: Train only the head (linear-probe style) when True.
        encoder_lr_mult: LR multiplier for the encoder when it is not frozen.
        head_type: ``"linear"`` or ``"mlp"``.
        head_hidden: Hidden width of the MLP head.
        head_dropout: Dropout inside the head.
        token_pool: Which tokens feed the head: ``"mean"``, ``"last"``,
            ``"mean_last"`` or ``"all"`` (flattened).
        out_scale_mgdl: Head outputs are residuals around the last observed
            glucose, scaled by this many mg/dL.
        window_stride: Training-window stride (steps) for DataFrame inputs.
    """

    # Set from ``_EXTRA`` in ``__init__``; declared here for type checkers
    pretrained_subfolder: Optional[str]
    pretrained_repo_id: str
    pretrained_local_dir: Optional[str]
    freeze_encoder: bool
    encoder_lr_mult: float
    head_type: str
    head_hidden: int
    head_dropout: float
    token_pool: str
    out_scale_mgdl: float
    window_stride: int
    max_train_windows: Optional[int]
    seed: int

    _EXTRA = {
        "pretrained_subfolder": "cgm_jepa",
        "pretrained_repo_id": "CRUISEResearchGroup/CGM-JEPA",
        "pretrained_local_dir": None,
        "freeze_encoder": True,
        "encoder_lr_mult": 0.1,
        "head_type": "mlp",
        "head_hidden": 256,
        "head_dropout": 0.1,
        "token_pool": "mean_last",
        "out_scale_mgdl": 50.0,
        "window_stride": 6,
        "max_train_windows": None,
        "seed": 42,
    }

    def __init__(self, **kwargs: Any) -> None:
        valid_base = {f.name for f in fields(ModelConfig)}
        base_kwargs = {
            k: v for k, v in kwargs.items() if k in valid_base and k != "model_type"
        }
        # Pretrained encoder is pinned to 24 h @ 5 min = 288 steps.
        base_kwargs.setdefault("context_length", 288)
        base_kwargs.setdefault("forecast_length", 24)
        base_kwargs.setdefault("learning_rate", 1e-3)
        base_kwargs.setdefault("batch_size", 128)
        base_kwargs.setdefault("num_epochs", 20)
        base_kwargs.setdefault("weight_decay", 1e-2)
        base_kwargs.setdefault("loss_function", "pinball")
        base_kwargs.setdefault("training_mode", "fine_tune")
        base_kwargs.setdefault("training_backend", TrainingBackend.PYTORCH)
        if isinstance(base_kwargs["training_backend"], str):
            base_kwargs["training_backend"] = TrainingBackend(
                base_kwargs["training_backend"]
            )
        super().__init__(**base_kwargs)
        self.model_type = "cgm_jepa"
        for key, default in self._EXTRA.items():
            setattr(self, key, kwargs.get(key, default))
        # The workflow's standard ``model_path`` names the HF repo
        if "pretrained_repo_id" not in kwargs and self.model_path:
            self.pretrained_repo_id = self.model_path

        if self.head_type not in ("linear", "mlp"):
            raise ValueError(
                f"head_type must be 'linear' or 'mlp', got {self.head_type}"
            )
        if self.token_pool not in ("mean", "last", "mean_last", "all"):
            raise ValueError(f"unknown token_pool: {self.token_pool}")
        if self.loss_function not in ("pinball", "mse", "mae"):
            raise ValueError(f"unsupported loss_function: {self.loss_function}")

    @property
    def quantiles_or_default(self) -> List[float]:
        return list(
            self.quantile_levels or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        )

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({k: getattr(self, k) for k in self._EXTRA})
        out["model_type"] = "cgm_jepa"
        return out

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "CgmJepaConfig":
        d = dict(config_dict)
        d.pop("model_type", None)
        return cls(**d)


def create_default_cgm_jepa_config(
    context_length: int = 288,
    forecast_length: int = 24,
    pretrained_subfolder: Optional[str] = "cgm_jepa",
    **overrides: Any,
) -> CgmJepaConfig:
    return CgmJepaConfig(
        context_length=context_length,
        forecast_length=forecast_length,
        pretrained_subfolder=pretrained_subfolder,
        **overrides,
    )
