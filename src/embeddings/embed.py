"""
Embedding generation: convert chunk text into vectors via an embedding API.

Provider is pluggable (EMBEDDING_PROVIDER in .env): "gemini" (default),
"openai", or "fake" (dry runs, no API call).

Default is Google's gemini-embedding-001 -- see
docs/adr/004-switch-to-gemini-embeddings.md for why: OpenAI billing hit a
real payment obstacle (cards declined, consistent with Indian banks
disabling international transactions by default), and Gemini's free tier
requires no credit card at all. This also isn't just a workaround --
gemini-embedding-001 supports task-type-aware embeddings (RETRIEVAL_DOCUMENT
vs RETRIEVAL_QUERY), which is a genuine retrieval-quality improvement over
OpenAI's symmetric embeddings, since a query and the document that answers
it are often phrased very differently. OpenAI's text-embedding-3-small
remains supported (see docs/adr/003-embedding-provider-choice.md) and Voyage
AI is a planned addition for a future retrieval-quality experiment.
Swapping providers is a config change, not a rewrite, since everything
downstream only depends on embed_texts() returning a list of vectors.

A "fake" provider is also available (EMBEDDING_PROVIDER=fake) for testing
the pipeline end-to-end without spending API credits: it produces
deterministic pseudo-embeddings from a hash of the text, so identical text
always gets the identical vector, and different text gets different
vectors -- good enough to exercise the wiring, not for real retrieval
quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from src.embeddings.vector_store import VectorStore

load_dotenv()

logger = logging.getLogger("embed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DEFAULT_BATCH_SIZE = 100
FAKE_EMBEDDING_DIM = 256


def _fake_embed_one(text: str) -> list[float]:
    """Deterministic pseudo-embedding for dry runs, with no API call."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    # Repeat the 32-byte digest to fill the target dimension, then scale
    # bytes (0-255) into a small float range.
    raw = (digest * (FAKE_EMBEDDING_DIM // len(digest) + 1))[:FAKE_EMBEDDING_DIM]
    return [(b - 128) / 128 for b in raw]


def _embed_batch_openai(texts: list[str], model: str) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI()  # reads OPENAI_API_KEY from the environment
    response = client.embeddings.create(input=texts, model=model)
    # OpenAI's API doesn't guarantee response order matches input order in
    # its documented contract, but each item carries its own `index` --
    # sort by that rather than trusting list order, to be safe.
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def _embed_batch_gemini(texts: list[str], model: str, is_query: bool) -> list[list[float]]:
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
    response = client.models.embed_content(
        model=model, contents=texts, config=types.EmbedContentConfig(task_type=task_type)
    )
    return [embedding.values for embedding in response.embeddings]


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """Embed a list of texts, batching requests to stay within API limits.

    is_query distinguishes search queries from indexed documents. Only
    Gemini uses this (RETRIEVAL_QUERY vs RETRIEVAL_DOCUMENT task types) --
    OpenAI and the fake provider have no task-type concept and ignore it.
    """
    if not texts:
        return []

    provider = os.getenv("EMBEDDING_PROVIDER", "gemini")
    batch_size = DEFAULT_BATCH_SIZE

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]

        if provider == "fake":
            batch_embeddings = [_fake_embed_one(t) for t in batch]
        elif provider == "openai":
            model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
            batch_embeddings = _embed_batch_openai(batch, model)
        elif provider == "gemini":
            model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
            batch_embeddings = _embed_batch_gemini(batch, model, is_query)
        else:
            raise ValueError(
                f"Unknown EMBEDDING_PROVIDER '{provider}'. Use 'gemini', 'openai', or 'fake'."
            )

        all_embeddings.extend(batch_embeddings)
        logger.info(
            "Embedded batch %d-%d of %d (provider=%s)", i, i + len(batch), len(texts), provider
        )

    return all_embeddings


def embed_chunks(chunks_path: Path) -> VectorStore:
    """Read chunks.jsonl, embed each chunk's text, and build a VectorStore."""
    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    if not chunks:
        logger.warning("No chunks found in %s", chunks_path)
        return VectorStore()

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    store = VectorStore()
    for chunk, vector in zip(chunks, vectors):
        metadata = {
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"],
            "source_feed": chunk["source_feed"],
            "title": chunk["title"],
            "reference_number": chunk.get("reference_number"),
            "master_direction_refs": chunk.get("master_direction_refs", []),
            "pub_date": chunk.get("pub_date"),
            "source_url": chunk.get("source_url", ""),
        }
        store.add(chunk["chunk_id"], vector, metadata)

    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks and build a vector store.")
    parser.add_argument(
        "--input", default="data/processed/chunks.jsonl", help="Path to chunks.jsonl."
    )
    parser.add_argument(
        "--output-dir", default="data/embeddings", help="Directory to write the vector store."
    )
    args = parser.parse_args()

    store = embed_chunks(Path(args.input))
    store.save(Path(args.output_dir))


if __name__ == "__main__":
    main()
