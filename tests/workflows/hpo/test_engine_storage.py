# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Optuna storage-backend selection in the HPO engine."""

import pytest

optuna = pytest.importorskip("optuna")

from optuna.storages import JournalStorage, RDBStorage  # noqa: E402

from src.workflows.hpo.engine import (  # noqa: E402
    _build_storage,
    _resolve_journal_path,
)


def test_resolve_journal_path_from_sqlite_url():
    # Legacy sqlite URLs map to a sibling .journal so existing configs keep working.
    assert str(_resolve_journal_path("sqlite:///results/optuna/x.db")) == (
        "results/optuna/x.journal"
    )


def test_resolve_journal_path_from_plain_and_scheme():
    assert str(_resolve_journal_path("results/optuna/x.journal")) == (
        "results/optuna/x.journal"
    )
    # journal:/// mirrors sqlite:///'s convention: 3 slashes -> relative path,
    # 4 slashes -> absolute path.
    assert str(_resolve_journal_path("journal:///tmp/y.journal")) == "tmp/y.journal"
    assert str(_resolve_journal_path("journal:////tmp/y.journal")) == "/tmp/y.journal"


def test_default_backend_is_journal(tmp_path):
    db = tmp_path / "study.db"
    storage = _build_storage({"storage": f"sqlite:///{db}"})
    assert isinstance(storage, JournalStorage)
    # The journal file is created alongside, with a .journal suffix.
    assert (tmp_path / "study.journal").exists()
    assert not db.exists()


def test_journal_backend_round_trips(tmp_path):
    config = {"storage": str(tmp_path / "s.journal"), "storage_backend": "journal"}
    study = optuna.create_study(
        study_name="rt", storage=_build_storage(config), load_if_exists=True
    )
    study.add_trial(
        optuna.trial.create_trial(
            params={"x": 1.0},
            distributions={"x": optuna.distributions.FloatDistribution(0.0, 2.0)},
            value=0.5,
        )
    )
    reopened = optuna.load_study(study_name="rt", storage=_build_storage(config))
    assert len(reopened.trials) == 1


def test_rdb_backend_opt_in(tmp_path):
    db = tmp_path / "study.db"
    storage = _build_storage({"storage": f"sqlite:///{db}", "storage_backend": "rdb"})
    assert isinstance(storage, RDBStorage)
    assert db.exists()


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="storage_backend"):
        _build_storage({"storage": "sqlite:///x.db", "storage_backend": "bogus"})
