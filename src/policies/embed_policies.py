"""
Embeds the synthetic policy corpus into its own vector store, kept separate
from the RBI regulation store (data/embeddings/) so the two corpora can be
searched independently or cross-referenced deliberately -- searching "which
policy is closest to this regulation" is exactly the cross-corpus query
src.analysis.policy_mapper performs.

Reuses chunk_document() and embed_texts() unchanged rather than duplicating
chunking/embedding logic for a second corpus.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.chunking.chunk import chunk_document
from src.embeddings.embed import embed_texts
from src.embeddings.vector_store import VectorStore
from src.policies.policy_loader import DEFAULT_POLICIES_DIR, load_policy_documents


def embed_policies(policies_dir: Path = DEFAULT_POLICIES_DIR) -> VectorStore:
    documents = load_policy_documents(policies_dir)

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    if not all_chunks:
        return VectorStore()

    texts = [c.text for c in all_chunks]
    vectors = embed_texts(texts, is_query=False)

    store = VectorStore()
    for chunk, vector in zip(all_chunks, vectors):
        metadata = {
            "document_id": chunk.document_id,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "source_feed": chunk.source_feed,
            "title": chunk.title,
            "reference_number": chunk.reference_number,
            "master_direction_refs": chunk.master_direction_refs,
            "pub_date": chunk.pub_date,
            "source_url": chunk.source_url,
        }
        store.add(chunk.chunk_id, vector, metadata)

    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed the synthetic policy corpus.")
    parser.add_argument(
        "--policies-dir",
        default=str(DEFAULT_POLICIES_DIR),
        help="Directory of policy markdown files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/embeddings/policies",
        help="Directory to write the vector store.",
    )
    args = parser.parse_args()

    store = embed_policies(Path(args.policies_dir))
    store.save(Path(args.output_dir))


if __name__ == "__main__":
    main()
