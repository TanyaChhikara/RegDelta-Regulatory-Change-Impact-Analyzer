"""Tests for src.preprocessing.preprocess."""

import json

from src.preprocessing.preprocess import (
    detect_entity_categories,
    load_raw_documents,
    process_document,
    save_processed_documents,
)

SAMPLE_RAW_RECORD = {
    "document_id": "abc123",
    "source_feed": "notifications",
    "title": "Reserve Bank of India (Commercial Banks) Directions, 2026",
    "reference_number": "RBI/2026-27/226",
    "master_direction_refs": ["13003"],
    "pub_date": "Fri, 31 Jul 2026 12:00:00",
    "source_url": "https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=13641",
    "raw_html": "<p>RBI/2026-27/226</p><p>Directions applicable to Commercial Banks.</p>",
    "clean_text": "unused -- process_document re-derives this from raw_html",
    "fetched_at": "2026-08-29T12:00:00+00:00",
}


def test_detect_entity_categories_finds_known_category():
    assert "Commercial Banks" in detect_entity_categories("Applicable to Commercial Banks.")


def test_detect_entity_categories_returns_empty_for_no_match():
    assert detect_entity_categories("No entity type mentioned here.") == []


def test_detect_entity_categories_matches_nbfc_variants():
    assert "NBFC" in detect_entity_categories("Applicable to Non-Banking Financial Companies.")
    assert "NBFC" in detect_entity_categories("Applicable to NBFCs.")


def test_process_document_reextracts_clean_text_from_raw_html():
    doc = process_document(SAMPLE_RAW_RECORD)
    # clean_text should come from raw_html via the table-aware extractor,
    # not the (possibly table-naive) clean_text already present in the raw record.
    assert doc.clean_text == "RBI/2026-27/226\nDirections applicable to Commercial Banks."


def test_process_document_carries_forward_metadata():
    doc = process_document(SAMPLE_RAW_RECORD)
    assert doc.document_id == "abc123"
    assert doc.reference_number == "RBI/2026-27/226"
    assert doc.master_direction_refs == ["13003"]
    assert "Commercial Banks" in doc.entity_categories


def test_process_document_computes_word_count():
    doc = process_document(SAMPLE_RAW_RECORD)
    assert doc.word_count == len(doc.clean_text.split())
    assert doc.word_count > 0


def test_load_raw_documents_deduplicates_by_document_id(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # Same document_id written to two different files, simulating overlap
    # between repeated fetches on different days.
    with open(raw_dir / "fetch_1.jsonl", "w") as f:
        f.write(json.dumps(SAMPLE_RAW_RECORD) + "\n")
    with open(raw_dir / "fetch_2.jsonl", "w") as f:
        f.write(json.dumps(SAMPLE_RAW_RECORD) + "\n")

    loaded = load_raw_documents(raw_dir)
    assert len(loaded) == 1


def test_load_raw_documents_returns_empty_list_when_dir_empty(tmp_path):
    raw_dir = tmp_path / "empty_raw"
    raw_dir.mkdir()
    assert load_raw_documents(raw_dir) == []


def test_save_processed_documents_writes_valid_jsonl(tmp_path):
    doc = process_document(SAMPLE_RAW_RECORD)
    output_path = save_processed_documents([doc], tmp_path / "processed")

    assert output_path.exists()
    lines = output_path.read_text().splitlines()
    assert len(lines) == 1

    loaded_back = json.loads(lines[0])
    assert loaded_back["document_id"] == "abc123"
