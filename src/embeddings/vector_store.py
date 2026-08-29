"""
A small, custom vector store: exact cosine-similarity search over an
in-memory NumPy array, with save/load to disk.

Why not Qdrant yet: see docs/adr/002-numpy-vector-store-before-qdrant.md.
Short version -- at ~20-100 vectors, brute-force search is both simpler to
understand and faster in practice than standing up a database built for
millions of vectors. The interface here (add/search/save/load) is
deliberately small so swapping in a real vector database later means
changing the implementation behind this interface, not the retrieval code
that calls it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("vector_store")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Scale a vector to unit length.

    Once every stored vector has unit length, cosine similarity between two
    vectors is just their dot product -- no need to divide by norms at
    search time, which keeps search a single fast matrix-vector multiply.
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


class VectorStore:
    """Exact cosine-similarity search over a small in-memory vector index."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.metadata: list[dict] = []
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None  # built lazily, invalidated on add()

    def add(self, id_: str, vector: list[float] | np.ndarray, metadata: dict) -> None:
        """Add one vector (and its metadata) to the store."""
        arr = np.asarray(vector, dtype=np.float32)
        self.ids.append(id_)
        self.metadata.append(metadata)
        self._vectors.append(_normalize(arr))
        self._matrix = None  # stale; rebuilt on next search

    def __len__(self) -> int:
        return len(self.ids)

    def _ensure_matrix(self) -> np.ndarray:
        if self._matrix is None:
            if not self._vectors:
                self._matrix = np.zeros((0, 0), dtype=np.float32)
            else:
                self._matrix = np.vstack(self._vectors)
        return self._matrix

    def search(self, query_vector: list[float] | np.ndarray, top_k: int = 5) -> list[dict]:
        """Return the top_k most similar stored vectors to `query_vector`.

        Each result is {"id": ..., "score": cosine_similarity, "metadata": ...},
        sorted by score descending.
        """
        if len(self) == 0:
            return []

        query = _normalize(np.asarray(query_vector, dtype=np.float32))
        matrix = self._ensure_matrix()
        scores = matrix @ query  # cosine similarity, since everything is unit-normalized

        top_k = min(top_k, len(self))
        # argpartition is O(n) vs argsort's O(n log n); fine either way at
        # this scale, but this is the pattern to keep if the corpus grows.
        top_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [
            {"id": self.ids[i], "score": float(scores[i]), "metadata": self.metadata[i]}
            for i in top_indices
        ]

    def save(self, output_dir: Path) -> None:
        """Persist the store to disk: vectors.npy + metadata.jsonl."""
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix = self._ensure_matrix()
        np.save(output_dir / "vectors.npy", matrix)

        with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for id_, meta in zip(self.ids, self.metadata):
                f.write(json.dumps({"id": id_, "metadata": meta}, ensure_ascii=False) + "\n")

        logger.info("Saved vector store (%d vectors) to %s", len(self), output_dir)

    @classmethod
    def load(cls, input_dir: Path) -> VectorStore:
        """Load a store previously written by save()."""
        store = cls()
        matrix = np.load(input_dir / "vectors.npy")

        with (input_dir / "metadata.jsonl").open(encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                store.ids.append(record["id"])
                store.metadata.append(record["metadata"])

        store._vectors = list(matrix)
        store._matrix = matrix if matrix.size else None

        logger.info("Loaded vector store (%d vectors) from %s", len(store), input_dir)
        return store
