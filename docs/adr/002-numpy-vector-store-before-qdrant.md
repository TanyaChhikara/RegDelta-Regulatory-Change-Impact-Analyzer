# ADR-002: Custom NumPy Vector Store Before Qdrant

**Status:** Accepted
**Date:** 2026-08-30

## Context

`.env.example` (from M0) already anticipates Qdrant as the vector database
(`QDRANT_HOST`, `QDRANT_PORT`), following the original technology shortlist.
At the point of building the first embedding/retrieval pipeline (M5), the
actual corpus is ~20 documents, each producing exactly one chunk (per M4's
findings) -- roughly 20 vectors total.

## Decision

**Build a small custom NumPy-based vector store for the MVP, and defer
Qdrant until corpus size or production concerns actually justify it.**

The store normalizes vectors to unit length at insert time and computes
cosine similarity via a single dot product against all stored vectors --
brute-force exact search, no approximate nearest-neighbor indexing.

## Reasoning

1. **Scale doesn't justify it yet.** Qdrant (like any real vector database)
   is built to make approximate nearest-neighbor search fast across millions
   of vectors. Brute-force cosine similarity over ~20-100 vectors runs in
   microseconds in plain NumPy. Standing up a database (and a Docker
   container to run it) for a workload this small adds operational
   complexity with no retrieval-quality or performance benefit.

2. **Pedagogical value.** This project's own stated philosophy is to
   implement something ourselves first when it teaches the underlying
   concept, and only reach for a framework once it provides real value
   beyond what a simple implementation already gives. Vector similarity
   search is exactly this kind of concept -- writing the cosine similarity
   computation directly is more instructive than calling
   `qdrant_client.search()` and trusting it works.

3. **No lock-in cost.** The store's interface (`add`, `search`, `save`,
   `load`) is intentionally small and Qdrant-shaped, so migrating later
   means swapping the implementation behind the same interface, not
   rewriting the retrieval pipeline that calls it.

## Alternatives Considered

| Option | Verdict |
|---|---|
| Qdrant (Docker) | Rejected for now -- real infrastructure for a 20-vector corpus is premature |
| FAISS | Similar verdict to Qdrant: built for scale we don't have yet, plus a compiled dependency to manage for no current benefit |
| pgvector | Requires a running Postgres instance; same "infrastructure before it's needed" concern |
| Custom NumPy store | **Chosen** -- zero infrastructure, transparent, fast enough at this scale |

## Consequences

- `data/embeddings/` will contain a `.npy` file (vectors) plus a `.jsonl`
  side-file (ids + metadata), not a Qdrant collection.
- Revisit this decision once either: (a) the corpus grows past a few
  thousand chunks (where brute-force search starts to matter), or (b) we
  need Qdrant-specific features this project will eventually want anyway --
  metadata filtering at scale, hybrid search integration, or persistence
  across restarts beyond simple file save/load.
- `.env.example`'s `QDRANT_HOST`/`QDRANT_PORT` remain unused for now; not
  removed, since Qdrant is still the intended destination once justified.
