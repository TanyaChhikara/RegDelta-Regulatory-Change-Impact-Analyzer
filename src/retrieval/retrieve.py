"""
Retrieval pipeline: the reusable interface between "a query string" and
"ranked, structured results" -- wrapping query embedding + vector search
behind one call.

This is Level 1 (basic vector retrieval) in the project's staged RAG
progression. Later levels (hybrid search, reranking, query rewriting,
agentic retrieval) are expected to sit behind this same Retriever.retrieve()
interface, so callers -- an evaluation script, an API, eventually an agent --
don't need to change when the retrieval strategy underneath gets more
sophisticated.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.embeddings.embed import embed_texts
from src.embeddings.vector_store import VectorStore

DEFAULT_TOP_K = 5


@dataclass
class RetrievalResult:
    """One retrieved chunk, with enough metadata to cite and inspect it."""

    chunk_id: str
    score: float
    document_id: str
    title: str
    text: str
    reference_number: str | None
    master_direction_refs: list[str]
    pub_date: str | None
    source_url: str


class Retriever:
    """Wraps a VectorStore with query embedding to answer natural-language queries."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    @classmethod
    def from_dir(cls, embeddings_dir: Path) -> Retriever:
        """Load a Retriever from a directory previously written by VectorStore.save()."""
        return cls(VectorStore.load(embeddings_dir))

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievalResult]:
        """Embed `query` and return the top_k most relevant chunks, ranked by score."""
        query_vector = embed_texts([query], is_query=True)[0]
        raw_results = self.vector_store.search(query_vector, top_k=top_k)
        return [self._to_result(r) for r in raw_results]

    @staticmethod
    def _to_result(raw: dict) -> RetrievalResult:
        meta = raw["metadata"]
        return RetrievalResult(
            chunk_id=raw["id"],
            score=raw["score"],
            document_id=meta["document_id"],
            title=meta["title"],
            text=meta["text"],
            reference_number=meta.get("reference_number"),
            master_direction_refs=meta.get("master_direction_refs", []),
            pub_date=meta.get("pub_date"),
            source_url=meta.get("source_url", ""),
        )


def _print_results(results: list[RetrievalResult]) -> None:
    if not results:
        print("No results.")
        return

    for rank, result in enumerate(results, start=1):
        print(f"\n[{rank}] score={result.score:.3f}  {result.title}")
        if result.reference_number:
            print(f"    ref: {result.reference_number}")
        if result.master_direction_refs:
            print(f"    links to Master Direction(s): {', '.join(result.master_direction_refs)}")
        snippet = result.text[:200] + ("..." if len(result.text) > 200 else "")
        print(f"    {snippet}")
        print(f"    {result.source_url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RegDelta vector store.")
    parser.add_argument("--query", required=True, help="Natural-language query.")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results.")
    parser.add_argument(
        "--embeddings-dir", default="data/embeddings", help="Directory holding the vector store."
    )
    args = parser.parse_args()

    retriever = Retriever.from_dir(Path(args.embeddings_dir))
    results = retriever.retrieve(args.query, top_k=args.top_k)
    _print_results(results)


if __name__ == "__main__":
    main()
