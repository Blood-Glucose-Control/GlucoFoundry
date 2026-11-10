# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``glucofoundry new-experiment`` scaffolder (Phase 4b)."""

from __future__ import annotations

import json
import py_compile
from pathlib import Path

import pytest
import yaml

from src.cli import main as cli_main
from src.cli import new_experiment as ne

_ANSWERS = {
    "experiment_name": "demo_exp",
    "experiment_title": "Demo experiment",
    "research_question": "Does scaffolding work?",
    "task_family": "forecasting",
    "target_col": "bg_mM",
    "context_length": 256,
    "forecast_length": 48,
    "sampling_interval_minutes": 5,
    "datasets": ["aleppo_2017", "brown_2019"],
    "modality": "unimodal",
    "anchoring_strategy": "sliding_window",
    "covariates": ["iob", "cob"],
    "models": ["ttm", "chronos2"],
    "metrics": ["rmse", "mae"],
    "holdout_strategy": "patient_level",
    "split_ratios": "0.1",
    "include_sweeps": True,
    "include_notebooks": True,
    "license": "Apache-2.0",
    "citation": "",
}


@pytest.fixture
def manifest() -> dict:
    return ne.load_manifest()


# --- answer validation ------------------------------------------------------
def test_validate_fills_defaults_and_coerces(manifest: dict) -> None:
    answers = ne.validate_answers(
        manifest,
        {"experiment_name": "x", "experiment_title": "X", "context_length": "128"},
    )
    assert answers["experiment_name"] == "x"
    assert answers["context_length"] == 128  # coerced to int
    assert answers["task_family"] == "forecasting"  # default
    assert answers["metrics"] == ["rmse", "mae"]  # default list


def test_validate_requires_name(manifest: dict) -> None:
    with pytest.raises(ne.AnswerError):
        ne.validate_answers(manifest, {})


@pytest.mark.parametrize("bad", ["Bad-Name", "1leading", "with space", "UPPER"])
def test_validate_rejects_bad_slug(manifest: dict, bad: str) -> None:
    with pytest.raises(ne.AnswerError):
        ne.validate_answers(manifest, {"experiment_name": bad})


def test_validate_rejects_bad_choice(manifest: dict) -> None:
    with pytest.raises(ne.AnswerError):
        ne.validate_answers(
            manifest,
            {"experiment_name": "x", "experiment_title": "X", "task_family": "nope"},
        )


def test_validate_rejects_unknown_multichoice(manifest: dict) -> None:
    with pytest.raises(ne.AnswerError):
        ne.validate_answers(
            manifest,
            {
                "experiment_name": "x",
                "experiment_title": "X",
                "models": ["not_a_model"],
            },
        )


# --- derived tokens / rendering --------------------------------------------
def test_derive_tokens() -> None:
    derived = ne.derive_tokens(
        {"experiment_name": "meal_image", "task_family": "forecasting"}
    )
    assert derived["summarizer_class"] == "MealImageSummarizer"
    assert derived["experiment_type"] == "meal_image_forecasting"


def test_render_text_unknown_token_raises() -> None:
    with pytest.raises(ne.AnswerError):
        ne.render_text("{{ nope }}", {"experiment_name": "x"})


def test_format_lists_and_bools() -> None:
    assert ne._format(["a", "b"]) == "a, b"
    assert ne._format([]) == "—"
    assert ne._format(True) == "true"


# --- generation -------------------------------------------------------------
def test_render_experiment_tree_and_wiring(tmp_path: Path) -> None:
    dest = ne.render_experiment(dict(_ANSWERS), tmp_path)
    assert dest == tmp_path / "demo_exp"
    for rel in [
        "register.py",
        "episodes.py",
        "eval.py",
        "summarize.py",
        "__init__.py",
        "README.md",
        "results/README.md",
        "trained_models/README.md",
    ]:
        assert (dest / rel).is_file(), rel
    # optional dirs present when requested
    assert (dest / "configs/sweeps/.gitkeep").is_file()
    assert (dest / "notebooks/.gitkeep").is_file()
    # rendered recipe compiles
    for py in dest.glob("*.py"):
        py_compile.compile(str(py), doraise=True)
    # register.py is wired with the derived summarizer + name
    register = (dest / "register.py").read_text()
    assert 'name="demo_exp"' in register
    assert "DemoExpSummarizer" in register
    assert ".tmpl" not in {p.suffix for p in dest.rglob("*")}


def test_render_experiment_omits_optional_dirs(tmp_path: Path) -> None:
    answers = {**_ANSWERS, "include_sweeps": False, "include_notebooks": False}
    dest = ne.render_experiment(answers, tmp_path)
    assert not (dest / "configs/sweeps").exists()
    assert not (dest / "notebooks").exists()
    assert (
        dest / "configs/models/.gitkeep"
    ).is_file()  # non-optional dir still present


def test_render_experiment_existing_dir(tmp_path: Path) -> None:
    ne.render_experiment(dict(_ANSWERS), tmp_path)
    with pytest.raises(FileExistsError):
        ne.render_experiment(dict(_ANSWERS), tmp_path)
    # --force overwrites
    ne.render_experiment(dict(_ANSWERS), tmp_path, force=True)


# --- answers file / CLI entry ----------------------------------------------
@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_load_answers_file(tmp_path: Path, suffix: str) -> None:
    path = tmp_path / f"answers{suffix}"
    path.write_text(
        json.dumps(_ANSWERS) if suffix == ".json" else yaml.safe_dump(_ANSWERS)
    )
    assert ne.load_answers_file(path)["experiment_name"] == "demo_exp"


def test_cli_main_from_file(tmp_path: Path) -> None:
    answers_file = tmp_path / "answers.yaml"
    answers_file.write_text(yaml.safe_dump(_ANSWERS))
    out = tmp_path / "out"
    rc = cli_main.main(
        ["new-experiment", "--from-file", str(answers_file), "--output-dir", str(out)]
    )
    assert rc == 0
    assert (out / "demo_exp" / "register.py").is_file()


def test_run_interview_scripted(manifest: dict) -> None:
    """Interactive path: scripted prompts exercise defaults, numbered + list picks."""

    def prompt_fn(text: str) -> str:
        if text.startswith("Experiment name"):
            return "scripted_exp"
        if text.startswith("One-line title"):
            return "Scripted title"
        if text.startswith("Task family"):
            return "1"  # numbered choice -> first option
        if text.startswith("Models to evaluate"):
            return "ttm, chronos2"  # multichoice by name
        return ""  # empty -> fall back to each question's default

    answers = ne.run_interview(manifest, prompt_fn)
    assert answers["experiment_name"] == "scripted_exp"
    assert answers["experiment_title"] == "Scripted title"
    assert answers["task_family"] == "forecasting"  # first numbered choice
    assert answers["models"] == ["ttm", "chronos2"]
    assert answers["target_col"] == "bg_mM"  # default via empty input


def test_require_templates_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ne, "templates_dir", lambda: tmp_path
    )  # no experiment.yaml here
    with pytest.raises(ne.ScaffolderError):
        ne.require_templates()


def test_next_steps_message_non_forecasting(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    answers_file = tmp_path / "answers.yaml"
    answers_file.write_text(
        yaml.safe_dump({**_ANSWERS, "task_family": "event_detection"})
    )
    out = tmp_path / "out"
    cli_main.main(
        ["new-experiment", "--from-file", str(answers_file), "--output-dir", str(out)]
    )
    captured = capsys.readouterr().out
    assert "forecasting workflow" not in captured
    assert "event_detection workflow" in captured
