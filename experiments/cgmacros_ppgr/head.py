# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""The PPGR head: an MLP with two hidden layers on frozen features.

The paper only says "the same MLP with two hidden layers"; everything in
``HeadConfig`` is our choice.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit


@dataclass(frozen=True)
class HeadConfig:
    hidden: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 300
    patience: int = 20
    val_fraction: float = 0.2  # share of training participants for early stopping


def _mlp(n_in: int, n_out: int, cfg: HeadConfig) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_in, cfg.hidden),
        nn.GELU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.hidden, cfg.hidden),
        nn.GELU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.hidden, n_out),
    )


def fit_predict_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    cfg: HeadConfig = HeadConfig(),
) -> np.ndarray:
    """Train on (x_train, y_train) and return predictions for x_test.

    Features and targets are standardized with training statistics. Training
    stops when the loss on held-out training participants stops improving.
    """
    torch.manual_seed(seed)
    x_mu, x_sd = x_train.mean(0), x_train.std(0) + 1e-8
    y_mu, y_sd = y_train.mean(0), y_train.std() + 1e-8

    def xt(x):
        return torch.tensor((x - x_mu) / x_sd, dtype=torch.float32)

    def yt(y):
        return torch.tensor((y - y_mu) / y_sd, dtype=torch.float32)

    split = GroupShuffleSplit(n_splits=1, test_size=cfg.val_fraction, random_state=seed)
    fit_idx, val_idx = next(split.split(x_train, groups=groups_train))
    xf, yf = xt(x_train[fit_idx]), yt(y_train[fit_idx])
    xv, yv = xt(x_train[val_idx]), yt(y_train[val_idx])

    net = _mlp(x_train.shape[1], y_train.shape[1], cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = np.inf, copy.deepcopy(net.state_dict()), 0
    for _ in range(cfg.max_epochs):
        net.train()
        for idx in torch.randperm(len(xf), generator=gen).split(cfg.batch_size):
            opt.zero_grad()
            loss = ((net(xf[idx]) - yf[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            val = float(((net(xv) - yv) ** 2).mean())
        if val < best - 1e-6:
            best, best_state, bad = val, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(xt(x_test)).numpy()
    return pred * y_sd + y_mu
