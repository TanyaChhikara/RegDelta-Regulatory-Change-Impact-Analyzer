"""Tests for src.analysis.analyze_regulation."""

import json

from src.analysis.analyze_regulation import load_regulation_by_reference


def test_load_regulation_by_reference_finds_matching_document(tmp_path):
    path = tmp_path / "processed_documents.jsonl"
    docs = [
        {"reference_number": "RBI/2026-27/1", "title": "A", "clean_text": "text a"},
        {"reference_number": "RBI/2026-27/2", "title": "B", "clean_text": "text b"},
    ]
    with path.open("w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    result = load_regulation_by_reference("RBI/2026-27/2", path)
    assert result is not None
    assert result["title"] == "B"


def test_load_regulation_by_reference_returns_none_when_not_found(tmp_path):
    path = tmp_path / "processed_documents.jsonl"
    with path.open("w") as f:
        f.write(json.dumps({"reference_number": "RBI/2026-27/1", "title": "A"}) + "\n")

    result = load_regulation_by_reference("RBI/2026-27/999", path)
    assert result is None


def test_load_regulation_by_reference_handles_documents_without_reference_number(tmp_path):
    path = tmp_path / "processed_documents.jsonl"
    docs = [
        {"reference_number": None, "title": "Press release", "clean_text": "text"},
        {"reference_number": "RBI/2026-27/1", "title": "A", "clean_text": "text a"},
    ]
    with path.open("w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    result = load_regulation_by_reference("RBI/2026-27/1", path)
    assert result is not None
    assert result["title"] == "A"
