"""
Preprocessing pipeline: raw fetched RBI documents -> normalized processed
documents ready for downstream chunking/embedding.

Reads every data/raw/*.jsonl file (as produced by src.ingestion.rbi_fetcher),
deduplicates by document_id, re-extracts clean_text using the table-aware
extractor (fixing the flattened-table problem the plain extractor has), tags
each document with the regulated-entity categories it mentions, and writes
one normalized record per document to data/processed/.

Why re-extract clean_text here rather than reuse the fetcher's version:
raw_html is preserved specifically so extraction logic can be improved after
the fact without re-fetching from RBI. This run re-derives clean_text from
raw_html using extract_clean_text_table_aware() instead of trusting the
fetcher's original (table-naive) clean_text field.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.preprocessing.text_cleaning import extract_clean_text_table_aware

logger = logging.getLogger("preprocess")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Regulated-entity categories from RBI's 2025 Master Direction consolidation
# (11 categories of regulated entities). Simple keyword detection -- not a
# classifier -- good enough to tag which categories a document touches.
ENTITY_CATEGORIES: dict[str, str] = {
    "Commercial Banks": r"Commercial Banks?",
    "Small Finance Banks": r"Small Finance Banks?",
    "Payments Banks": r"Payments? Banks?",
    "Regional Rural Banks": r"Regional Rural Banks?",
    "Local Area Banks": r"Local Area Banks?",
    "Urban Co-operative Banks": r"Urban Co-?operative Banks?",
    "Rural Co-operative Banks": r"Rural Co-?operative Banks?",
    "NBFC": r"Non-?Banking Financial Compan(?:y|ies)|NBFCs?",
    "All India Financial Institutions": r"All India Financial Institutions?",
    "Credit Information Companies": r"Credit Information Compan(?:y|ies)",
    "Asset Reconstruction Companies": r"Asset Reconstruction Compan(?:y|ies)",
}
_ENTITY_PATTERNS = {
    name: re.compile(pattern, re.IGNORECASE) for name, pattern in ENTITY_CATEGORIES.items()
}


@dataclass
class ProcessedDocument:
    """A normalized, cleaned RBI document ready for chunking/embedding."""

    document_id: str
    source_feed: str
    title: str
    reference_number: str | None
    master_direction_refs: list[str]
    pub_date: str | None
    source_url: str
    clean_text: str
    word_count: int
    entity_categories: list[str]
    fetched_at: str
    processed_at: str


def detect_entity_categories(text: str) -> list[str]:
    """Return which regulated-entity categories are mentioned in `text`."""
    return [name for name, pattern in _ENTITY_PATTERNS.items() if pattern.search(text)]


def load_raw_documents(raw_dir: Path) -> list[dict]:
    """Load and deduplicate every raw document from data/raw/*.jsonl."""
    files = sorted(glob.glob(str(raw_dir / "*.jsonl")))
    if not files:
        logger.warning("No .jsonl files found in %s", raw_dir)

    records: list[dict] = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                records.append(json.loads(line))

    seen_ids: set[str] = set()
    deduped: list[dict] = []
    for record in records:
        doc_id = record.get("document_id")
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        deduped.append(record)

    logger.info(
        "Loaded %d raw documents from %d file(s), %d after dedup",
        len(records), len(files), len(deduped),
    )
    return deduped


def process_document(raw: dict) -> ProcessedDocument:
    """Convert one raw fetched-document record into a ProcessedDocument."""
    clean_text = extract_clean_text_table_aware(raw["raw_html"])
    title = raw.get("title", "")

    return ProcessedDocument(
        document_id=raw["document_id"],
        source_feed=raw["source_feed"],
        title=title,
        reference_number=raw.get("reference_number"),
        master_direction_refs=raw.get("master_direction_refs", []),
        pub_date=raw.get("pub_date"),
        source_url=raw.get("source_url", ""),
        clean_text=clean_text,
        word_count=len(clean_text.split()),
        entity_categories=detect_entity_categories(f"{title} {clean_text}"),
        fetched_at=raw.get("fetched_at", ""),
        processed_at=datetime.now(timezone.utc).isoformat(),
    )


def process_all(raw_dir: Path) -> list[ProcessedDocument]:
    raw_documents = load_raw_documents(raw_dir)
    return [process_document(raw) for raw in raw_documents]


def save_processed_documents(documents: list[ProcessedDocument], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "processed_documents.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

    logger.info("Saved %d processed documents to %s", len(documents), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess raw RBI documents into normalized, cleaned records."
    )
    parser.add_argument(
        "--raw-dir", default="data/raw", help="Directory of raw fetched JSONL files."
    )
    parser.add_argument(
        "--output-dir", default="data/processed", help="Directory to write processed output."
    )
    args = parser.parse_args()

    documents = process_all(Path(args.raw_dir))
    save_processed_documents(documents, Path(args.output_dir))


if __name__ == "__main__":
    main()
