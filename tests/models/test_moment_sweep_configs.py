"""Lightweight tests for MOMENT sweep configs and probabilistic contract."""

import numpy as np
import pandas as pd
import pytest

from src.models.moment.config import MomentConfig
from src.models.moment.model import MomentForecaster


class _DummyMomentPipeline:
    """Minimal stand-in for MOMENTPipeline used in unit tests."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def init(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


@pytest.fixture
def patched_moment_import(monkeypatch):
    from src.models.moment import model as moment_model

    monkeypatch.setattr(
        moment_model, "_optional_moment_import", lambda: _DummyMomentPipeline
    )
    return moment_model


def test_moment_declares_non_probabilistic(patched_moment_import):
    model = MomentForecaster(
        MomentConfig(
            model_path="AutonLab/MOMENT-1-small",
            context_length=512,
            forecast_length=72,
            training_mode="zero_shot",
            use_cpu=True,
        )
    )
    assert model.supports_probabilistic_forecast is False


def test_moment_quantile_request_raises(patched_moment_import):
    model = MomentForecaster(
        MomentConfig(
            model_path="AutonLab/MOMENT-1-small",
            context_length=512,
            forecast_length=72,
            training_mode="zero_shot",
            use_cpu=True,
        )
    )

    with pytest.raises(NotImplementedError, match="does not support probabilistic"):
        model.predict(
            pd.DataFrame({"bg_mM": np.linspace(6.0, 8.0, 520)}),
            quantile_levels=[0.1, 0.5, 0.9],
        )


def test_moment_predict_batch_chunks(monkeypatch, patched_moment_import):
    model = MomentForecaster(
        MomentConfig(
            model_path="AutonLab/MOMENT-1-small",
            context_length=16,
            forecast_length=4,
            training_mode="zero_shot",
            batch_size=2,
            use_cpu=True,
        )
    )

    call_sizes = []

    def _fake_forecast_batch(contexts, prediction_length, context_lengths=None):
        call_sizes.append(contexts.shape[0])
        return np.full((contexts.shape[0], prediction_length), 7.0, dtype=np.float32)

    monkeypatch.setattr(model, "_forecast_batch", _fake_forecast_batch)

    panel = []
    for i in range(5):
        ep = pd.DataFrame(
            {
                "episode_id": [f"ep_{i}"] * 20,
                "bg_mM": np.linspace(6.0 + i, 8.0 + i, 20),
            }
        )
        panel.append(ep)
    panel_df = pd.concat(panel, ignore_index=True)

    preds = model.predict_batch(panel_df, episode_col="episode_id")
    assert len(preds) == 5
    assert sorted(call_sizes) == [1, 2, 2]
    for v in preds.values():
        assert v.shape == (4,)


def test_moment_predict_batch_with_covariates(monkeypatch, patched_moment_import):
    model = MomentForecaster(
        MomentConfig(
            model_path="AutonLab/MOMENT-1-small",
            context_length=16,
            forecast_length=4,
            training_mode="zero_shot",
            batch_size=2,
            use_cpu=True,
            covariate_cols=["iob"],
        )
    )

    seen_shapes = []

    def _fake_forecast_batch(contexts, prediction_length, context_lengths=None):
        seen_shapes.append(tuple(contexts.shape))
        return np.full((contexts.shape[0], prediction_length), 7.0, dtype=np.float32)

    monkeypatch.setattr(model, "_forecast_batch", _fake_forecast_batch)

    panel = []
    for i in range(3):
        ep = pd.DataFrame(
            {
                "episode_id": [f"ep_{i}"] * 20,
                "bg_mM": np.linspace(6.0 + i, 8.0 + i, 20),
                "iob": np.linspace(0.1, 0.5, 20),
            }
        )
        panel.append(ep)
    panel_df = pd.concat(panel, ignore_index=True)

    preds = model.predict_batch(panel_df, episode_col="episode_id")
    assert len(preds) == 3
    # contexts shape: [batch, time, channels] with target+covariate => 2 channels
    assert all(len(s) == 3 and s[2] == 2 for s in seen_shapes)
