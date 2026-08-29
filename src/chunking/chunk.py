"""
Chunking: split processed documents into bounded-size chunks for embedding.

EDA on the raw corpus (notebooks/01-raw-rbi-data-eda.ipynb) suggested most
RBI notifications and press releases are short enough that per-document
retrieval might not even require splitting. This module measures that with
real token counts (not the word-count proxy EDA used) and only splits a
document if it actually exceeds the target chunk size -- most documents in
the current corpus are expected to come back as a single chunk. The
paragraph-aware splitter with overlap exists for when longer documents (full
Master Directions, not just the amendment notifications fetched so far) are
ingested and genuinely need it.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken

logger = logging.getLogger("chunking")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# cl100k_base is the tokenizer used by OpenAI's text-embedding-3-* models
# (and gpt-3.5/gpt-4) -- using the real tokenizer here, not a word-count
# approximation, since that's what actually determines how much of a
# document an embedding model "sees" per chunk.
#
# tiktoken downloads this encoding's data file over the network on first use
# and caches it locally. On a restricted network (corporate firewall,
# air-gapped CI runner) that download can fail -- in that case we fall back
# to a word-count-based approximation rather than crashing the whole
# pipeline. The approximation is less accurate but keeps chunking decisions
# roughly sane until real network access is available.
def _load_encoding() -> tiktoken.Encoding | None:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: any failure here should degrade, not crash
        logger.warning(
            "Could not load tiktoken's cl100k_base encoding (%s). Falling back to a "
            "word-count-based token approximation; chunk boundaries will be "
            "less precise until this environment has network access to download it.",
            exc,
        )
        return None


_ENCODING = _load_encoding()

# Rule-of-thumb ratio for English text when the real tokenizer is unavailable.
_FALLBACK_TOKENS_PER_WORD = 1.3


def count_tokens(text: str) -> int:
    """Count tokens using the same tokenizer OpenAI's embedding models use.

    Falls back to a word-count approximation if the tokenizer's encoding
    data couldn't be loaded (see _load_encoding).
    """
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return int(len(text.split()) * _FALLBACK_TOKENS_PER_WORD)


DEFAULT_TARGET_CHUNK_TOKENS = 500
DEFAULT_CHUNK_OVERLAP_TOKENS = 50


@dataclass
class Chunk:
    """A bounded-size piece of a document, ready for embedding."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    # Metadata carried over from the parent document -- useful for retrieval
    # filtering and for citing back to the source document later.
    source_feed: str
    title: str
    reference_number: str | None
    master_direction_refs: list[str]
    pub_date: str | None
    source_url: str


def split_into_paragraphs(text: str) -> list[str]:
    """Split clean_text into its natural line units.

    Each line in clean_text is already a paragraph, table row, or heading --
    that structure comes from the preprocessing step (M3) -- so splitting on
    single newlines respects the document's own boundaries rather than
    cutting mid-sentence the way fixed-character splitting would.
    """
    return [line for line in text.split("\n") if line.strip()]


def pack_paragraphs_into_chunks(
    paragraphs: list[str],
    target_tokens: int = DEFAULT_TARGET_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Greedily pack paragraphs into chunks up to `target_tokens` each.

    When a chunk fills up, the trailing paragraphs of that chunk (up to
    `overlap_tokens` worth) are carried over as the start of the next chunk,
    so a concept that spans a chunk boundary isn't lost to either side.
    """
    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens > target_tokens and current_paragraphs:
            chunks.append("\n".join(current_paragraphs))

            # Carry trailing paragraphs into the next chunk as overlap,
            # walking backward from the end until we'd exceed the budget.
            overlap_paragraphs: list[str] = []
            overlap_count = 0
            for p in reversed(current_paragraphs):
                p_tokens = count_tokens(p)
                if overlap_count + p_tokens > overlap_tokens:
                    break
                overlap_paragraphs.insert(0, p)
                overlap_count += p_tokens

            current_paragraphs = overlap_paragraphs
            current_tokens = overlap_count

        current_paragraphs.append(para)
        current_tokens += para_tokens

    if current_paragraphs:
        chunks.append("\n".join(current_paragraphs))

    return chunks


def chunk_document(
    doc: dict,
    target_tokens: int = DEFAULT_TARGET_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk one processed document (as a dict, e.g. loaded from JSONL)."""
    text = doc["clean_text"]
    total_tokens = count_tokens(text)

    if total_tokens <= target_tokens:
        chunk_texts = [text] if text else []
    else:
        paragraphs = split_into_paragraphs(text)
        chunk_texts = pack_paragraphs_into_chunks(paragraphs, target_tokens, overlap_tokens)

    chunks = []
    for i, chunk_text in enumerate(chunk_texts):
        chunks.append(
            Chunk(
                chunk_id=f"{doc['document_id']}_chunk{i}",
                document_id=doc["document_id"],
                chunk_index=i,
                text=chunk_text,
                token_count=count_tokens(chunk_text),
                source_feed=doc["source_feed"],
                title=doc["title"],
                reference_number=doc.get("reference_number"),
                master_direction_refs=doc.get("master_direction_refs", []),
                pub_date=doc.get("pub_date"),
                source_url=doc.get("source_url", ""),
            )
        )
    return chunks


def chunk_all(
    processed_path: Path,
    target_tokens: int = DEFAULT_TARGET_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk every document in a processed_documents.jsonl file."""
    all_chunks: list[Chunk] = []
    multi_chunk_docs = 0
    total_docs = 0

    with open(processed_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            total_docs += 1
            doc_chunks = chunk_document(doc, target_tokens, overlap_tokens)
            if len(doc_chunks) > 1:
                multi_chunk_docs += 1
            all_chunks.extend(doc_chunks)

    logger.info(
        "Chunked %d documents into %d chunks (%d documents needed >1 chunk, target=%d tokens)",
        total_docs, len(all_chunks), multi_chunk_docs, target_tokens,
    )
    return all_chunks


def save_chunks(chunks: list[Chunk], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "chunks.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    logger.info("Saved %d chunks to %s", len(chunks), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk processed RBI documents for embedding.")
    parser.add_argument(
        "--input",
        default="data/processed/processed_documents.jsonl",
        help="Path to processed_documents.jsonl.",
    )
    parser.add_argument(
        "--output-dir", default="data/processed", help="Directory to write chunks.jsonl."
    )
    parser.add_argument(
        "--target-tokens", type=int, default=DEFAULT_TARGET_CHUNK_TOKENS,
        help="Target chunk size in tokens.",
    )
    parser.add_argument(
        "--overlap-tokens", type=int, default=DEFAULT_CHUNK_OVERLAP_TOKENS,
        help="Overlap between consecutive chunks, in tokens.",
    )
    args = parser.parse_args()

    chunks = chunk_all(Path(args.input), args.target_tokens, args.overlap_tokens)
    save_chunks(chunks, Path(args.output_dir))


if __name__ == "__main__":
    main()
