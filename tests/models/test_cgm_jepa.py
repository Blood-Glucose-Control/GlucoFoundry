"""Offline tests for the CGM-JEPA encoder + quantile-head forecaster.

All tests use a randomly initialised encoder (``pretrained_subfolder=None``) so
they never touch the network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models.cgm_jepa import CgmJepaConfig, CgmJepaForecaster  # noqa: E402
from src.models.cgm_jepa.model import (  # noqa: E402
    make_windows,
    pinball_loss,
    prepare_contexts,
)

CTX, HOR = 96, 12


def _cfg(**kw) -> CgmJepaConfig:
    base = dict(
        context_length=CTX,
        forecast_length=HOR,
        pretrained_subfolder=None,
        num_epochs=3,
        batch_size=32,
        learning_rate=3e-3,
        use_cpu=True,
        head_hidden=32,
    )
    base.update(kw)
    return CgmJepaConfig(**base)


def _synthetic(n: int = 256, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(CTX + HOR)
    phase = rng.uniform(0, 2 * np.pi, (n, 1))
    series = (
        140 + 40 * np.sin(2 * np.pi * t / 60 + phase) + rng.normal(0, 2, (n, t.size))
    )
    return series[:, :CTX].astype(np.float32), series[:, CTX:].astype(np.float32)


def test_prepare_contexts_pads_and_fills_nan():
    out = prepare_contexts([np.array([100.0, np.nan, 110.0]), np.full(500, 90.0)], CTX)
    assert out.shape == (2, CTX)
    assert np.isfinite(out).all()
    assert out[0, 0] == 100.0 and out[0, -1] == 110.0  # left-pad with first value
    assert out[0, -2] == 100.0  # NaN forward-filled


def test_context_length_must_match_patch():
    with pytest.raises(ValueError, match="multiple"):
        CgmJepaForecaster(_cfg(context_length=100))


def test_make_windows_drops_nonfinite():
    s = np.arange(200, dtype=np.float32)
    s[50] = np.nan
    x, y = make_windows([s], 100, 10, stride=10)
    assert x.shape[1] == 100 and y.shape[1] == 10
    assert np.isfinite(x).all() and np.isfinite(y).all()


def test_pinball_matches_manual():
    pred = torch.tensor([[[1.0], [3.0]]])  # (B=1, Q=2, H=1)
    target = torch.tensor([[2.0]])
    q = torch.tensor([0.1, 0.9])
    # q=.1: err=+1 -> .1 ; q=.9: err=-1 -> .1 ; mean = .1
    assert pinball_loss(pred, target, q).item() == pytest.approx(0.1)


def test_fit_reduces_loss_and_freezes_encoder(tmp_path):
    model = CgmJepaForecaster(_cfg(freeze_encoder=True))
    assert model.net is not None
    before = {k: v.clone() for k, v in model.net.encoder.state_dict().items()}
    x, y = _synthetic()
    metrics = model.fit((x, y), output_dir=str(tmp_path))
    hist = metrics["training_history"]["train_loss"]
    assert hist[-1] < hist[0]
    after = model.net.encoder.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # encoder untouched


def test_finetune_updates_encoder(tmp_path):
    model = CgmJepaForecaster(_cfg(freeze_encoder=False, num_epochs=2))
    assert model.net is not None
    w0 = model.net.encoder.encoder_norm.weight.clone()
    model.net.encoder.encoder_norm.weight.data += 0.0  # no-op, keep reference
    x, y = _synthetic(64)
    model.fit((x, y), output_dir=str(tmp_path))
    assert not torch.equal(w0, model.net.encoder.encoder_norm.weight)


def test_predict_arrays_shapes_and_monotone_median_access(tmp_path):
    model = CgmJepaForecaster(_cfg(quantile_levels=[0.1, 0.5, 0.9]))
    x, y = _synthetic(64)
    model.fit((x, y), output_dir=str(tmp_path))
    point = model.predict_arrays(list(x[:5]))
    assert point.shape == (5, 1, HOR)
    qs = model.predict_arrays(list(x[:5]), [0.1, 0.5, 0.9])
    assert qs.shape == (5, 3, HOR)
    np.testing.assert_allclose(qs[:, 1:2], point, atol=1e-5)
    with pytest.raises(ValueError, match="not trained"):
        model.predict_arrays(list(x[:2]), [0.25])


def test_dataframe_api_uses_mmol_units(tmp_path):
    model = CgmJepaForecaster(_cfg())
    x, y = _synthetic(64)
    model.fit((x, y), output_dir=str(tmp_path))
    frame = pd.DataFrame({"bg_mM": x[0] / 18.018})
    pred = model.predict(frame)
    assert pred.shape == (HOR,)
    assert 2.0 < float(np.mean(pred)) < 20.0  # plausible mmol/L, not mg/dL
    q = model.predict(frame, quantile_levels=[0.1, 0.5, 0.9])
    assert q.shape == (3, HOR)


def test_save_load_roundtrip(tmp_path):
    model = CgmJepaForecaster(_cfg())
    x, y = _synthetic(64)
    model.fit((x, y), output_dir=str(tmp_path))
    expected = model.predict_arrays(list(x[:4]))
    model.save(str(tmp_path))
    loaded = CgmJepaForecaster.load(str(tmp_path))
    assert loaded.is_fitted
    np.testing.assert_allclose(loaded.predict_arrays(list(x[:4])), expected, atol=1e-5)
