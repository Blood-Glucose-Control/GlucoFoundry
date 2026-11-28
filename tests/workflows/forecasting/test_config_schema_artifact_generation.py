"""Tests for JSON schema artifact generation."""

from __future__ import annotations

import json

from src.config.schemas.json_schema_artifacts import (
    SCHEMA_ARTIFACT_SPECS,
    generate_json_schema_artifacts,
)


def test_generate_json_schema_artifacts_writes_expected_files(tmp_path) -> None:
    written = generate_json_schema_artifacts(tmp_path)

    expected_filenames = {spec.filename for spec in SCHEMA_ARTIFACT_SPECS}
    expected_filenames.add("manifest.json")
    expected_filenames.add("index.md")

    assert {path.name for path in written} == expected_filenames
    for filename in expected_filenames:
        artifact_path = tmp_path / filename
        assert artifact_path.exists()
        if filename.endswith(".json"):
            json.loads(artifact_path.read_text())

    manifest_payload = json.loads((tmp_path / "manifest.json").read_text())
    manifest_files = [entry["file"] for entry in manifest_payload["schemas"]]
    assert manifest_files == [spec.filename for spec in SCHEMA_ARTIFACT_SPECS]


def test_generated_index_lists_every_schema_file(tmp_path) -> None:
    generate_json_schema_artifacts(tmp_path)

    index_text = (tmp_path / "index.md").read_text()
    assert "manifest.json" in index_text
    for spec in SCHEMA_ARTIFACT_SPECS:
        assert spec.filename in index_text
