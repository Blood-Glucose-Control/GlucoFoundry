# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""CGM-JEPA encoder + quantile forecast head.

CGM-JEPA (Muhammad et al., 2026) is a self-supervised representation model, so
it ships no forecasting head. This wrapper pairs the pre-trained 24 h encoder
with a small trainable head that predicts ``forecast_length`` future steps as
quantile residuals around the last observed glucose.

Two operating modes (``config.freeze_encoder``):

* frozen encoder -> only the head trains (linear-probe), or
* fine-tune -> encoder trains at ``learning_rate * encoder_lr_mult``.

Units: the encoder was pre-trained on raw mg/dL, so the net works in mg/dL.
:meth:`fit` takes either ``(X, Y)`` mg/dL arrays or a DataFrame with ``bg_mM``;
:meth:`predict` takes ``bg_mM`` and returns mmol/L (:meth:`predict_arrays` is the
mg/dL version). Conversion happens at the boundary.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ...data.models import ColumnNames
from ...utils.logging_helper import info_print
from ..base import BaseTimeSeriesFoundationModel, ModelRegistry, TrainingBackend
from .config import MGDL_PER_MMOL, CgmJepaConfig
from .encoder import CgmJepaEncoder

_CHECKPOINT_NAME = "cgm_jepa_net.pt"


def pinball_loss(
    pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """Pinball loss. ``pred``: (B, Q, H); ``target``: (B, H); ``quantiles``: (Q,)."""
    err = target.unsqueeze(1) - pred
    q = quantiles.view(1, -1, 1)
    return torch.maximum(q * err, (q - 1.0) * err).mean()


class CgmJepaForecastNet(nn.Module):
    """Encoder + head. Input ``(B, C)`` mg/dL -> ``(B, Q, H)`` mg/dL."""

    def __init__(
        self,
        encoder: CgmJepaEncoder,
        context_length: int,
        forecast_length: int,
        n_quantiles: int,
        head_type: str = "mlp",
        head_hidden: int = 256,
        head_dropout: float = 0.1,
        token_pool: str = "mean_last",
        out_scale_mgdl: float = 50.0,
    ) -> None:
        super().__init__()
        patch = encoder.patch_len
        if context_length % patch != 0:
            raise ValueError(
                f"context_length ({context_length}) must be a multiple of the "
                f"encoder patch length ({patch})."
            )
        self.encoder = encoder
        self.patch_len = patch
        self.n_patches = context_length // patch
        self.forecast_length = forecast_length
        self.n_quantiles = n_quantiles
        self.token_pool = token_pool
        self.out_scale = float(out_scale_mgdl)

        d = encoder.embed_dim
        feat = {"mean": d, "last": d, "mean_last": 2 * d, "all": d * self.n_patches}[
            token_pool
        ]
        feat += 3  # last value, 30-min delta, 1-h delta (normalised scalars)
        out = n_quantiles * forecast_length
        if head_type == "linear":
            self.head: nn.Module = nn.Linear(feat, out)
        else:
            self.head = nn.Sequential(
                nn.Linear(feat, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, out),
            )
        # Zero the output layer so an untrained head predicts the last value
        last_layer = self.head if isinstance(self.head, nn.Linear) else self.head[-1]
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)

    def _pool(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.token_pool == "mean":
            return tokens.mean(dim=1)
        if self.token_pool == "last":
            return tokens[:, -1]
        if self.token_pool == "mean_last":
            return torch.cat([tokens.mean(dim=1), tokens[:, -1]], dim=-1)
        return tokens.flatten(1)

    def forward(self, x: torch.Tensor, freeze_encoder: bool = False) -> torch.Tensor:
        b = x.size(0)
        patches = x.reshape(b, self.n_patches, self.patch_len)
        if freeze_encoder:
            with torch.no_grad():
                tokens = self.encoder(patches)
        else:
            tokens = self.encoder(patches)
        last = x[:, -1:]
        scalars = torch.cat(
            [
                (last - 140.0) / 60.0,
                (last - x[:, -7:-6]) / 30.0,
                (last - x[:, -13:-12]) / 40.0,
            ],
            dim=-1,
        )
        feats = torch.cat([self._pool(tokens), scalars], dim=-1)
        resid = self.head(feats).reshape(b, self.n_quantiles, self.forecast_length)
        return last.unsqueeze(1) + resid * self.out_scale


def prepare_contexts(contexts: Sequence[np.ndarray], context_length: int) -> np.ndarray:
    """Stack variable-length contexts into ``(N, context_length)``.

    Keeps the most recent ``context_length`` steps; shorter histories are
    left-padded with their first valid value; NaNs are forward/back-filled
    (all-NaN rows fall back to a 120 mg/dL constant).
    """
    out = np.empty((len(contexts), context_length), dtype=np.float32)
    for i, raw in enumerate(contexts):
        arr = pd.Series(np.asarray(raw, dtype=np.float64)[-context_length:])
        arr = arr.ffill().bfill()
        vals = arr.to_numpy()
        if not np.isfinite(vals).any():
            vals = np.full(max(len(vals), 1), 120.0)
        if len(vals) < context_length:
            vals = np.concatenate([np.full(context_length - len(vals), vals[0]), vals])
        out[i] = vals
    return out


def make_windows(
    series: Sequence[np.ndarray],
    context_length: int,
    forecast_length: int,
    stride: int,
    max_windows: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sliding (context, target) windows over mg/dL series; drops non-finite."""
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for s in series:
        s = np.asarray(s, dtype=np.float32)
        n = len(s) - context_length - forecast_length + 1
        if n <= 0:
            continue
        for start in range(0, n, stride):
            x = s[start : start + context_length]
            y = s[start + context_length : start + context_length + forecast_length]
            if np.isfinite(x).all() and np.isfinite(y).all():
                xs.append(x)
                ys.append(y)
    if not xs:
        return (
            np.empty((0, context_length), np.float32),
            np.empty((0, forecast_length), np.float32),
        )
    x_arr, y_arr = np.stack(xs), np.stack(ys)
    if max_windows is not None and len(x_arr) > max_windows:
        idx = np.random.default_rng(seed).choice(len(x_arr), max_windows, replace=False)
        x_arr, y_arr = x_arr[np.sort(idx)], y_arr[np.sort(idx)]
    return x_arr, y_arr


@ModelRegistry.register("cgm_jepa")
class CgmJepaForecaster(BaseTimeSeriesFoundationModel):
    """CGM-JEPA encoder + quantile head (see module docstring)."""

    config_class = CgmJepaConfig

    def __init__(self, config: CgmJepaConfig) -> None:
        self.net: Optional[CgmJepaForecastNet] = None
        self._device = torch.device("cpu")
        super().__init__(config)

    # -- properties --------------------------------------------------------
    @property
    def training_backend(self) -> TrainingBackend:
        return TrainingBackend.PYTORCH

    @property
    def supports_zero_shot(self) -> bool:
        return False  # the head must be trained

    @property
    def supports_probabilistic_forecast(self) -> bool:
        return True

    # -- construction ------------------------------------------------------
    def _build_encoder(self) -> CgmJepaEncoder:
        cfg: CgmJepaConfig = self.config  # type: ignore[assignment]
        if cfg.pretrained_subfolder:
            info_print(
                f"Loading CGM-JEPA encoder '{cfg.pretrained_subfolder}' from "
                f"{cfg.pretrained_local_dir or cfg.pretrained_repo_id}"
            )
            return CgmJepaEncoder.from_pretrained(
                cfg.pretrained_repo_id,
                cfg.pretrained_subfolder,
                local_dir=cfg.pretrained_local_dir,
            )
        info_print("Initialising CGM-JEPA encoder from scratch (no pretrained weights)")
        return CgmJepaEncoder()

    def _build_net(self, encoder: CgmJepaEncoder) -> CgmJepaForecastNet:
        cfg: CgmJepaConfig = self.config  # type: ignore[assignment]
        return CgmJepaForecastNet(
            encoder,
            context_length=cfg.context_length,
            forecast_length=cfg.forecast_length,
            n_quantiles=len(self._train_quantiles()),
            head_type=cfg.head_type,
            head_hidden=cfg.head_hidden,
            head_dropout=cfg.head_dropout,
            token_pool=cfg.token_pool,
            out_scale_mgdl=cfg.out_scale_mgdl,
        )

    def _train_quantiles(self) -> List[float]:
        cfg: CgmJepaConfig = self.config  # type: ignore[assignment]
        levels = sorted({round(float(q), 6) for q in cfg.quantiles_or_default} | {0.5})
        return levels

    def _resolve_device(self) -> torch.device:
        if self.config.use_cpu or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda")

    def _initialize_model(self) -> None:
        torch.manual_seed(int(getattr(self.config, "seed", 42)))
        self.net = self._build_net(self._build_encoder())
        self.model = self.net
        self._device = self._resolve_device()
        self.net.to(self._device)

    # -- data --------------------------------------------------------------
    def _frame_to_series(self, data: pd.DataFrame) -> List[np.ndarray]:
        col = ColumnNames.BG.value
        if col not in data.columns:
            raise ValueError(f"DataFrame needs a '{col}' column")
        if "episode_id" in data.columns:
            groups = [
                g[col].to_numpy() for _, g in data.groupby("episode_id", sort=False)
            ]
        else:
            groups = [data[col].to_numpy()]
        return [np.asarray(g, dtype=np.float64) * MGDL_PER_MMOL for g in groups]

    def _prepare_training_data(
        self, train_data: Any
    ) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Return ``(train_loader, val_loader, None)``.

        ``train_data`` is either an ``(X, Y)`` tuple of mg/dL arrays or a
        DataFrame with ``bg_mM`` (and optional ``episode_id``). A
        ``val_fraction`` share is held out for early stopping: whole episodes
        for DataFrames (overlapping windows would leak), random windows for
        arrays. ``val_loader`` is ``None`` when ``val_fraction`` is 0.
        """
        cfg: CgmJepaConfig = self.config  # type: ignore[assignment]
        rng = np.random.default_rng(int(cfg.seed))
        val_fraction = float(cfg.val_fraction or 0.0)
        x_val = y_val = None
        if isinstance(train_data, tuple) and len(train_data) == 2:
            x, y = (np.asarray(a, dtype=np.float32) for a in train_data)
            x = prepare_contexts(list(x), cfg.context_length)
            if val_fraction > 0.0 and len(x) >= 4:
                order = rng.permutation(len(x))
                n_val = min(max(1, round(len(x) * val_fraction)), len(x) - 1)
                x_val, y_val = x[order[:n_val]], y[order[:n_val]]
                x, y = x[order[n_val:]], y[order[n_val:]]
        elif isinstance(train_data, pd.DataFrame):
            series = self._frame_to_series(train_data)
            val_series: List[np.ndarray] = []
            if val_fraction > 0.0 and len(series) >= 2:
                order = rng.permutation(len(series))
                n_val = min(max(1, round(len(series) * val_fraction)), len(series) - 1)
                val_series = [series[i] for i in order[:n_val]]
                series = [series[i] for i in order[n_val:]]
            x, y = make_windows(
                series,
                cfg.context_length,
                cfg.forecast_length,
                cfg.window_stride,
                cfg.max_train_windows,
                int(cfg.seed),
            )
            if val_series:
                x_val, y_val = make_windows(
                    val_series,
                    cfg.context_length,
                    cfg.forecast_length,
                    cfg.window_stride,
                )
        else:
            raise TypeError("train_data must be a (X, Y) tuple or a DataFrame")
        if len(x) == 0:
            raise ValueError("No training windows could be built from train_data")

        def _loader(xa: np.ndarray, ya: np.ndarray, shuffle: bool) -> DataLoader:
            ds = TensorDataset(
                torch.from_numpy(xa), torch.from_numpy(ya.astype(np.float32))
            )
            return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle)

        val_loader = None
        if x_val is not None and y_val is not None and len(x_val) > 0:
            val_loader = _loader(x_val, y_val, shuffle=False)
        return _loader(x, y, shuffle=True), val_loader, None

    # -- training ----------------------------------------------------------
    def _compute_loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_fn = self.config.loss_function
        if loss_fn == "pinball":
            q = torch.tensor(self._train_quantiles(), device=pred.device)
            return pinball_loss(pred, y, q)
        median = pred[:, self._train_quantiles().index(0.5)]
        if loss_fn == "mae":
            return (median - y).abs().mean()
        return ((median - y) ** 2).mean()

    def _train_model(
        self,
        train_data: Any,
        output_dir: str,
        epoch_callback: Optional[Callable[[int, float], None]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        cfg: CgmJepaConfig = self.config  # type: ignore[assignment]
        assert self.net is not None
        loader, val_loader, _ = self._prepare_training_data(train_data)
        net, device = self.net, self._device
        freeze = bool(cfg.freeze_encoder)
        for p in net.encoder.parameters():
            p.requires_grad = not freeze

        groups: List[Dict[str, Any]] = [
            {"params": list(net.head.parameters()), "lr": cfg.learning_rate}
        ]
        if not freeze:
            groups.append(
                {
                    "params": list(net.encoder.parameters()),
                    "lr": cfg.learning_rate * cfg.encoder_lr_mult,
                }
            )
        opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
        n_total = sum(p.numel() for p in net.parameters())
        n_train = sum(p.numel() for g in groups for p in g["params"])
        info_print(
            f"CGM-JEPA trainable params: {n_train / 1e3:.1f}k / {n_total / 1e3:.1f}k"
            f"; {len(loader.dataset)} train windows"  # type: ignore[arg-type]
            + (
                f", {len(val_loader.dataset)} val windows"  # type: ignore[arg-type]
                if val_loader is not None
                else ""
            )
        )

        patience = int(cfg.early_stopping_patience or 0)
        best_monitor = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_without_improve = 0
        losses: List[float] = []
        val_losses: List[float] = []
        for epoch in range(cfg.num_epochs):
            net.train()
            if freeze:
                net.encoder.eval()  # deterministic tokens
            tot, nb = 0.0, 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = self._compute_loss(net(xb, freeze_encoder=freeze), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in groups for p in g["params"]], cfg.gradient_clip_val
                )
                opt.step()
                tot += float(loss.item())
                nb += 1
            avg = tot / max(nb, 1)
            losses.append(avg)

            val_loss: Optional[float] = None
            if val_loader is not None:
                net.eval()
                vtot, vnb = 0.0, 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        vtot += float(self._compute_loss(net(xb, True), yb).item())
                        vnb += 1
                val_loss = vtot / max(vnb, 1)
                val_losses.append(val_loss)
            info_print(
                f"   Epoch {epoch + 1}/{cfg.num_epochs}: loss={avg:.4f}"
                + (f", val_loss={val_loss:.4f}" if val_loss is not None else "")
            )
            if epoch_callback is not None:
                epoch_callback(epoch, avg)

            monitor = val_loss if val_loss is not None else avg
            if np.isfinite(monitor) and monitor < best_monitor - 1e-6:
                best_monitor = monitor
                best_state = {
                    k: v.detach().cpu().clone() for k, v in net.state_dict().items()
                }
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1
                if patience and epochs_without_improve >= patience:
                    info_print(
                        f"   Early stopping at epoch {epoch + 1} "
                        f"(best monitor={best_monitor:.4f})"
                    )
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.is_fitted = True
        history: Dict[str, List[float]] = {"train_loss": losses}
        if val_losses:
            history["val_loss"] = val_losses
        return {
            "status": "success",
            "samples": len(loader.dataset),  # type: ignore[arg-type]
            "epochs": len(losses),
            "final_loss": losses[-1] if losses else float("nan"),
            "best_val_loss": best_monitor if val_losses else None,
            "training_history": history,
        }

    # -- inference ---------------------------------------------------------
    @torch.no_grad()
    def predict_arrays(
        self,
        contexts_mgdl: Sequence[np.ndarray],
        quantile_levels: Optional[Sequence[float]] = None,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Forecast mg/dL contexts -> ``(N, Q, H)`` (Q=1 median if no levels given)."""
        assert self.net is not None
        cfg: CgmJepaConfig = self.config  # type: ignore[assignment]
        trained = self._train_quantiles()
        wanted = (
            [0.5]
            if quantile_levels is None
            else [round(float(q), 6) for q in quantile_levels]
        )
        missing = [q for q in wanted if q not in trained]
        if missing:
            raise ValueError(
                f"quantile_levels {missing} were not trained; available: {trained}"
            )
        idx = [trained.index(q) for q in wanted]
        x = prepare_contexts(list(contexts_mgdl), cfg.context_length)
        self.net.eval()
        outs: List[np.ndarray] = []
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[i : i + batch_size]).to(self._device)
            # Sort along the quantile axis so quantiles never cross
            pred = self.net(xb, freeze_encoder=True).sort(dim=1).values
            outs.append(pred[:, idx].cpu().numpy())
        return (
            np.concatenate(outs, axis=0)
            if outs
            else np.empty((0, len(idx), cfg.forecast_length))
        )

    def _predict(
        self,
        data: pd.DataFrame,
        quantile_levels: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        del kwargs
        series = self._frame_to_series(data)[0]
        out = self.predict_arrays([series], quantile_levels)[0] / MGDL_PER_MMOL
        return out[0] if quantile_levels is None else out

    def _predict_batch(
        self,
        data: pd.DataFrame,
        episode_col: str,
        quantile_levels: Optional[List[float]] = None,
    ) -> Dict[str, np.ndarray]:
        col = ColumnNames.BG.value
        ids, ctxs = [], []
        for ep_id, g in data.groupby(episode_col, sort=False):
            ids.append(str(ep_id))
            ctxs.append(g[col].to_numpy(dtype=np.float64) * MGDL_PER_MMOL)
        preds = self.predict_arrays(ctxs, quantile_levels) / MGDL_PER_MMOL
        return {
            ep: (p[0] if quantile_levels is None else p) for ep, p in zip(ids, preds)
        }

    # -- checkpointing -----------------------------------------------------
    def _save_checkpoint(self, output_dir: str) -> None:
        assert self.net is not None
        os.makedirs(output_dir, exist_ok=True)
        enc = self.net.encoder
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "quantiles": self._train_quantiles(),
                "encoder_cfg": {
                    "dim_in": enc.dim_in,
                    "embed_dim": enc.embed_dim,
                    "nhead": enc.predictor_blocks[0].attn.num_heads,
                    "num_layers": len(enc.predictor_blocks),
                },
            },
            os.path.join(output_dir, _CHECKPOINT_NAME),
        )

    def _load_checkpoint(self, model_dir: str) -> None:
        blob = torch.load(
            os.path.join(model_dir, _CHECKPOINT_NAME),
            map_location="cpu",
            weights_only=True,
        )
        self.config.quantile_levels = list(blob["quantiles"])
        self.net = self._build_net(CgmJepaEncoder(**blob["encoder_cfg"]))
        self.net.load_state_dict(blob["state_dict"])
        self.model = self.net
        self.net.to(self._device)
        self.net.eval()
        self.is_fitted = True
