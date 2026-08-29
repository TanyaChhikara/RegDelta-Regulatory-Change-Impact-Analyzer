"""
Loads the synthetic policy corpus (data/synthetic/policies/*.md) into the
same document shape the rest of the pipeline already understands, so the
existing chunking and embedding code (src/chunking/chunk.py,
src/embeddings/embed.py) can be reused unchanged rather than duplicated for
a second corpus.

Policy files use a simple frontmatter block (key: value pairs between two
--- lines) rather than full YAML -- every value here is a plain scalar
string, so a hand-rolled parser avoids adding a YAML dependency for
something this small.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_POLICIES_DIR = Path("data/synthetic/policies")


def parse_frontmatter(raw_text: str) -> tuple[dict[str, str], str]:
    """Split a policy file into its frontmatter dict and body text.

    Expects the file to start with a '---' line, then 'key: value' lines,
    then a closing '---' line, then the body. Returns ({}, raw_text)
    unchanged if no frontmatter block is found, rather than raising --
    malformed frontmatter shouldn't crash the whole load.
    """
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw_text

    frontmatter: dict[str, str] = {}
    body_start = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body_start = i + 1
            break
        if ":" in line:
            key, _, value = line.partition(":")
            frontmatter[key.strip()] = value.strip()

    if body_start is None:
        # Opening '---' with no matching closing '---' -- treat as no
        # frontmatter rather than guessing.
        return {}, raw_text

    body = "\n".join(lines[body_start:]).strip()
    return frontmatter, body


def load_policy_manifest(policies_dir: Path = DEFAULT_POLICIES_DIR) -> list[dict]:
    with open(policies_dir / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def load_policy_documents(policies_dir: Path = DEFAULT_POLICIES_DIR) -> list[dict]:
    """Load every policy as a document dict shaped like a processed RBI
    document, so chunk_document() and embed_texts() can be reused as-is.

    Fields that don't apply to policies (reference_number,
    master_direction_refs) are set to sensible empty defaults rather than
    omitted, since downstream code accesses them via .get() with defaults
    anyway -- this keeps the shape uniform across both corpora.
    """
    manifest = load_policy_manifest(policies_dir)
    documents = []

    for entry in manifest:
        file_path = policies_dir / entry["file"]
        raw_text = file_path.read_text(encoding="utf-8")
        frontmatter, body = parse_frontmatter(raw_text)

        documents.append(
            {
                "document_id": entry["policy_id"],
                "source_feed": "synthetic_policy",
                "title": entry["title"],
                "reference_number": None,
                "master_direction_refs": [],
                "pub_date": frontmatter.get("last_reviewed"),
                "source_url": "",
                "clean_text": body,
                "entity_type": entry.get("entity_type", frontmatter.get("entity_type")),
                "expected_relevance": entry.get("expected_relevance"),
            }
        )

    return documents
