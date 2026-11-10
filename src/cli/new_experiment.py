# Copyright (c) 2025 Blood-Glucose-Control
# SPDX-License-Identifier: Apache-2.0

"""Scaffold a new experiment from ``templates/experiment/``.

Implements ``glucofoundry new-experiment``: a lightweight, dependency-free
generator (no cookiecutter) that drives the interview declared in
``templates/experiment.yaml``, then renders the ``templates/experiment/`` tree
into ``experiments/<name>/`` — replacing ``{{ token }}`` placeholders and
stripping the ``.tmpl`` suffix from rendered files.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import yaml

_TOKEN_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")
_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
def repo_root() -> Path:
    """Repository root (``src/cli/new_experiment.py`` → parents[2])."""
    return Path(__file__).resolve().parents[2]


def templates_dir() -> Path:
    return repo_root() / "templates"


def load_manifest() -> dict[str, Any]:
    """Load the interview manifest (``templates/experiment.yaml``)."""
    return yaml.safe_load((templates_dir() / "experiment.yaml").read_text())


# ---------------------------------------------------------------------------
# Dynamic choices (resolved at runtime so the manifest never goes stale)
# ---------------------------------------------------------------------------
def resolve_choices(source: str) -> list[str]:
    """Resolve a question's ``choices_source`` to a concrete option list."""
    if source == "dataset_loaders":
        from src.data.dataset_configs import list_available_datasets

        return sorted(list_available_datasets())
    if source == "supported_models":
        from src.models.factory import SUPPORTED_MODEL_TYPES

        return list(SUPPORTED_MODEL_TYPES)
    raise ValueError(f"unknown choices_source: {source!r}")


def question_choices(question: dict[str, Any]) -> list[str]:
    source = question.get("choices_source")
    if source:
        return resolve_choices(source)
    return list(question.get("choices", []))


# ---------------------------------------------------------------------------
# Answer validation / coercion
# ---------------------------------------------------------------------------
class AnswerError(ValueError):
    """Raised when provided answers are invalid or incomplete."""


def _coerce(question: dict[str, Any], value: Any) -> Any:
    key, qtype = question["key"], question["type"]
    if qtype == "slug":
        value = str(value).strip()
        if not _SLUG_RE.match(value):
            raise AnswerError(
                f"{key}: {value!r} is not a valid slug "
                "(lowercase letters/digits/underscores, must start with a letter)"
            )
        return value
    if qtype == "str":
        return str(value)
    if qtype == "int":
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise AnswerError(f"{key}: {value!r} is not an integer") from exc
    if qtype == "bool":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "yes", "y", "1"}
    if qtype == "choice":
        value = str(value)
        choices = question_choices(question)
        if value not in choices:
            raise AnswerError(f"{key}: {value!r} is not one of {choices}")
        return value
    if qtype == "multichoice":
        if isinstance(value, str):
            value = [v.strip() for v in value.split(",") if v.strip()]
        value = list(value)
        choices = question_choices(question)
        bad = [v for v in value if v not in choices]
        if bad:
            raise AnswerError(f"{key}: {bad} not in {choices}")
        return value
    raise AnswerError(f"{key}: unknown question type {qtype!r}")


def validate_answers(
    manifest: dict[str, Any], provided: dict[str, Any]
) -> dict[str, Any]:
    """Return a complete, type-coerced answer set (defaults filled in)."""
    answers: dict[str, Any] = {}
    for question in manifest["questions"]:
        key = question["key"]
        if key in provided and provided[key] not in (None, ""):
            answers[key] = _coerce(question, provided[key])
        elif "default" in question:
            answers[key] = _coerce(question, question["default"])
        elif question.get("required"):
            raise AnswerError(f"missing required answer: {key}")
        else:
            answers[key] = ""
    return answers


def derive_tokens(answers: dict[str, Any]) -> dict[str, Any]:
    """Compute tokens not asked directly (summarizer_class, experiment_type)."""
    name = answers["experiment_name"]
    summarizer = "".join(part.capitalize() for part in name.split("_")) + "Summarizer"
    return {
        "summarizer_class": summarizer,
        "experiment_type": f"{name}_{answers['task_family']}",
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _format(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) if value else "—"
    return str(value)


def render_text(text: str, tokens: dict[str, Any]) -> str:
    def _sub(match: re.Match[str]) -> str:
        key = match.group(1)
        if key not in tokens:
            raise AnswerError(f"template references undefined token {{{{ {key} }}}}")
        return _format(tokens[key])

    return _TOKEN_RE.sub(_sub, text)


def _skip(rel: Path, answers: dict[str, Any]) -> bool:
    parts = rel.parts
    if not answers.get("include_sweeps") and parts[:2] == ("configs", "sweeps"):
        return True
    if not answers.get("include_notebooks") and parts[:1] == ("notebooks",):
        return True
    return False


def render_experiment(
    answers: dict[str, Any],
    dest_root: Path,
    *,
    force: bool = False,
) -> Path:
    """Render ``templates/experiment/`` into ``dest_root/<name>/``."""
    name = answers["experiment_name"]
    dest = dest_root / name
    if dest.exists() and not force:
        raise FileExistsError(f"{dest} already exists (use --force to overwrite)")

    tokens = {**answers, **derive_tokens(answers)}
    src = templates_dir() / "experiment"
    for path in sorted(src.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(src)
        if _skip(rel, answers):
            continue
        out = dest / rel
        if path.suffix == ".tmpl":
            out = out.with_suffix("")  # strip the .tmpl suffix
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(render_text(path.read_text(), tokens))
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
    return dest


# ---------------------------------------------------------------------------
# Interactive interview
# ---------------------------------------------------------------------------
def _ask(question: dict[str, Any], prompt_fn: Callable[[str], str]) -> Any:
    key, qtype = question["key"], question["type"]
    default = question.get("default")
    label = question["prompt"]

    if qtype in {"choice", "multichoice"}:
        choices = question_choices(question)
        listing = "\n".join(f"  {i + 1}. {c}" for i, c in enumerate(choices))
        suffix = " (comma-separated)" if qtype == "multichoice" else ""
        raw = prompt_fn(
            f"{label}{suffix} [default: {_format(default)}]:\n{listing}\n> "
        )
        if not raw.strip():
            return question.get("default", [] if qtype == "multichoice" else "")
        picks = [p.strip() for p in raw.split(",") if p.strip()]
        resolved = [choices[int(p) - 1] if p.isdigit() else p for p in picks]
        return resolved if qtype == "multichoice" else resolved[0]

    raw = prompt_fn(f"{label} [default: {_format(default)}]: ")
    if not raw.strip():
        if default is None and question.get("required"):
            raise AnswerError(f"{key} is required")
        return default
    return raw.strip()


def run_interview(
    manifest: dict[str, Any], prompt_fn: Callable[[str], str] = input
) -> dict[str, Any]:
    raw = {q["key"]: _ask(q, prompt_fn) for q in manifest["questions"]}
    return validate_answers(manifest, raw)


# ---------------------------------------------------------------------------
# Answers file (non-interactive)
# ---------------------------------------------------------------------------
def load_answers_file(path: Path) -> dict[str, Any]:
    text = path.read_text()
    data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise AnswerError(f"{path}: expected a mapping of answers")
    return data


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "new-experiment",
        help="Scaffold a new experiment under experiments/<name>/.",
        description="Scaffold a new experiment from templates/experiment/.",
    )
    parser.add_argument(
        "--from-file",
        type=Path,
        default=None,
        help="YAML/JSON answers file for non-interactive scaffolding.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to create the experiment in (default: <repo>/experiments).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing experiment directory.",
    )
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    if args.from_file is not None:
        answers = validate_answers(manifest, load_answers_file(args.from_file))
    else:
        answers = run_interview(manifest)

    dest_root = args.output_dir or (repo_root() / "experiments")
    dest = render_experiment(answers, dest_root, force=args.force)

    name = answers["experiment_name"]
    print(f"\n✓ Scaffolded experiment '{name}' at {dest}")
    print("Next steps:")
    print(f"  1. Implement the stubs in {dest.name}/episodes.py, eval.py, summarize.py")
    print(f"  2. Add data/model configs under {dest.name}/configs/")
    print(f"  3. Run the forecasting workflow with --experiment {name}")
    return 0
