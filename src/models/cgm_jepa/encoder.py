# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""CGM-JEPA context encoder (vendored, inference/fine-tune path only).

Ported from https://github.com/cruiseresearchgroup/CGM-JEPA (MIT License,
Muhammad et al., 2026; arXiv:2605.00933). Only the encoder is vendored; the
JEPA predictor / pre-training code is not needed for forecasting. Module and
parameter names are kept identical to upstream so the published
``CRUISEResearchGroup/CGM-JEPA`` safetensors load with ``strict=True``.

The encoder consumes a 24 h CGM window at 5-minute resolution (288 steps,
raw mg/dL, *not* z-scored: upstream pre-trains with ``normalize_x=False``),
splits it into hourly patches of 12 steps, and returns one 96-d token per patch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

HF_REPO_ID = "CRUISEResearchGroup/CGM-JEPA"
# Subfolders published in the HF repo (the second adds a glucodensity objective).
PRETRAINED_SUBFOLDERS: Tuple[str, ...] = ("cgm_jepa", "x_cgm_jepa")


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class _MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,
        qk_scale: Optional[float],
        attn_drop: float,
        proj_drop: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = (
            self.qkv(x)
            .reshape(b, n, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = self.attn_drop(((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(out))


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        qk_scale: Optional[float],
        drop: float,
        attn_drop: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _MultiHeadAttention(
            dim, num_heads, qkv_bias, qk_scale, attn_drop, drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pos_emb = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_emb", pos_emb.unsqueeze(0))

    def forward(self, length: int) -> torch.Tensor:
        return self.pos_emb[:, :length]  # type: ignore[index]


class _ValueEmbedding(nn.Module):
    """Per-patch Conv1d (kernel=stride=``patch_size``'s ``kernel_size``) + linear."""

    def __init__(
        self, dim: int, in_channels: int, kernel_size: int, bias: bool
    ) -> None:
        super().__init__()
        self.proj = nn.Conv1d(
            1, dim, kernel_size=kernel_size, stride=kernel_size, bias=bias
        )
        conv_len = (in_channels - kernel_size) // kernel_size + 1
        self.fc = nn.Linear(dim * conv_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, p = x.shape  # (B, num_patches, patch_len)
        x = self.proj(x.reshape(b * n, 1, p)).reshape(b * n, -1)
        return self.fc(x).reshape(b, n, -1)


class _TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model: int, d_inp: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_inp, d_model)

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        b, n, p, d = x_mark.shape
        out = self.proj(x_mark.reshape(-1, d).float()).reshape(b, n, p, -1)
        return out.mean(dim=2)


class _DataEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        in_channels: int,
        kernel_size: int,
        embed_bias: bool,
        time_inp_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.value_embedding = _ValueEmbedding(
            dim, in_channels, kernel_size, embed_bias
        )
        self.positional_embedding = _PositionalEmbedding(dim)
        self.timefeature_embedding = _TimeFeatureEmbedding(dim, time_inp_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.value_embedding(x) + self.positional_embedding(x.size(1))
        return self.dropout(out)


class CgmJepaEncoder(nn.Module):
    """CGM-JEPA context encoder: ``(B, L, 12)`` patches -> ``(B, L, embed_dim)`` tokens.

    Upstream's optional timestamp features (``x_mark``) are not used here: the
    published downstream evaluation passes none, and the time-feature branch is
    a no-op in that case, so the parameters exist only for checkpoint parity.
    """

    def __init__(
        self,
        dim_in: int = 12,
        kernel_size: int = 3,
        embed_dim: int = 96,
        embed_bias: bool = True,
        nhead: int = 6,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        time_inp_dim: int = 5,
        jepa: bool = False,
    ) -> None:
        super().__init__()
        self.dim_in = dim_in  # patch length (steps per token)
        self.embed_dim = embed_dim
        self.jepa = jepa
        self.data_embedding = _DataEmbedding(
            embed_dim, dim_in, kernel_size, embed_bias, time_inp_dim, drop_rate
        )
        self.predictor_blocks = nn.ModuleList(
            [
                _Block(
                    embed_dim,
                    nhead,
                    mlp_ratio,
                    qkv_bias,
                    qk_scale,
                    drop_rate,
                    attn_drop_rate,
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)
        # Upstream projection head used by the JEPA pre-training loss.
        self.proj = _MLP(embed_dim, 1024, 48)

    @property
    def patch_len(self) -> int:
        return self.dim_in

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.data_embedding(patches)  # embed each hour + position code
        for block in self.predictor_blocks:  # 3 transformer layers
            x = block(x)
        return self.encoder_norm(x)  # final layer norm

    @classmethod
    def from_config_dict(cls, cfg: Dict[str, Any]) -> "CgmJepaEncoder":
        known = {
            "dim_in",
            "kernel_size",
            "embed_dim",
            "embed_bias",
            "nhead",
            "num_layers",
            "mlp_ratio",
            "qkv_bias",
            "qk_scale",
            "drop_rate",
            "attn_drop_rate",
            "time_inp_dim",
            "jepa",
        }
        return cls(**{k: v for k, v in cfg.items() if k in known})

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = HF_REPO_ID,
        subfolder: str = "cgm_jepa",
        local_dir: Optional[str] = None,
    ) -> "CgmJepaEncoder":
        """Load published weights (HF hub, or a local dir holding the subfolder)."""
        from safetensors.torch import load_file

        if local_dir is not None:
            base = Path(local_dir) / subfolder
            config_path, weights_path = base / "config.json", base / "model.safetensors"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(hf_hub_download(repo_id, f"{subfolder}/config.json"))
            weights_path = Path(
                hf_hub_download(repo_id, f"{subfolder}/model.safetensors")
            )
        with open(config_path, "r", encoding="utf-8") as fh:
            encoder = cls.from_config_dict(json.load(fh))
        encoder.load_state_dict(load_file(str(weights_path)), strict=True)
        return encoder
