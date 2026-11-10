# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Integrity tests for the `glucofoundry new-experiment` scaffold template.

These guard the template skeleton (Phase 4a) against drift before the scaffolder
CLI (Phase 4b) consumes it:

- every ``{{ token }}`` used in a template file is declared in the interview
  manifest (``templates/experiment.yaml``), and
- the rendered ``*.py.tmpl`` recipe stubs are valid, importable Python.
"""

from __future__ import annotations

import py_compile
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = REPO_ROOT / "templates"
EXPERIMENT_TMPL = TEMPLATES / "experiment"

_TOKEN_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

# Representative answers used only to validate that rendered stubs compile.
_SAMPLE_ANSWERS = {
    "experiment_name": "demo",
    "experiment_title": "Demo forecasting experiment",
    "research_question": "Can a demo be scaffolded?",
    "license": "Apache-2.0",
    "task_family": "forecasting",
    "target_col": "bg_mM",
    "context_length": "512",
    "forecast_length": "96",
    "sampling_interval_minutes": "5",
    "anchoring_strategy": "sliding_window",
    "metrics": "['rmse', 'mae']",
    "summarizer_class": "DemoSummarizer",
    "experiment_type": "demo_forecasting",
}


def _manifest() -> dict:
    return yaml.safe_load((TEMPLATES / "experiment.yaml").read_text())


def _declared_tokens() -> set[str]:
    manifest = _manifest()
    return {q["key"] for q in manifest["questions"]} | set(manifest.get("derived", {}))


def _render(text: str) -> str:
    return _TOKEN_RE.sub(lambda m: _SAMPLE_ANSWERS.get(m.group(1), m.group(1)), text)


def test_manifest_questions_are_well_formed() -> None:
    manifest = _manifest()
    assert manifest["questions"], "interview manifest declares no questions"
    keys = [q["key"] for q in manifest["questions"]]
    assert len(keys) == len(set(keys)), "duplicate question keys in manifest"
    for q in manifest["questions"]:
        assert q.get("key") and q.get("prompt") and q.get("type"), q


def test_all_template_tokens_are_declared() -> None:
    declared = _declared_tokens()
    unknown: dict[str, str] = {}
    for tmpl in EXPERIMENT_TMPL.rglob("*.tmpl"):
        for token in _TOKEN_RE.findall(tmpl.read_text()):
            if token not in declared:
                unknown[token] = str(tmpl.relative_to(TEMPLATES))
    assert not unknown, f"undeclared tokens in templates: {unknown}"


def test_rendered_python_stubs_compile(tmp_path: Path) -> None:
    stubs = sorted(EXPERIMENT_TMPL.rglob("*.py.tmpl"))
    assert stubs, "expected rendered .py.tmpl recipe stubs"
    for stub in stubs:
        out = tmp_path / stub.name[: -len(".tmpl")]
        out.write_text(_render(stub.read_text()))
        py_compile.compile(str(out), doraise=True)


@pytest.mark.parametrize(
    "required",
    ["register.py.tmpl", "episodes.py.tmpl", "eval.py.tmpl", "summarize.py.tmpl"],
)
def test_core_recipe_files_present(required: str) -> None:
    assert (EXPERIMENT_TMPL / required).is_file()
