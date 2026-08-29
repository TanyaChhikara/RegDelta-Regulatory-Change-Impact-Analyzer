"""
Cross-corpus retrieval: given a regulatory document's text, find candidate
internal policies that might be affected.

This is the actual point of connecting the two corpora built in M2-M7 --
regulations and policies are embedded with the same model into compatible
vector spaces, so a regulation's text can be used directly as a query
against the *policy* vector store. Retriever (from src.retrieval.retrieve)
is already generic enough to do this unchanged: it doesn't know or care
what corpus its vector store holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.retrieval.retrieve import Retriever

DEFAULT_POLICY_EMBEDDINGS_DIR = Path("data/embeddings/policies")
DEFAULT_TOP_K = 3

# How many raw chunk-level results to fetch before deduplicating to
# distinct documents. Some policies (per M8 part 1's finding) split into
# multiple chunks, so a plain top_k chunk fetch can return the same policy
# more than once, crowding out genuinely different candidates. Overfetching
# and deduplicating down to top_k distinct documents fixes this; found via
# a real run where a 3-chunk-document policy occupied 2 of 3 top-k slots.
OVERFETCH_FACTOR = 4


@dataclass
class PolicyMatch:
    """One candidate policy retrieved as potentially relevant to a regulation."""

    policy_id: str
    title: str
    score: float
    text: str


def find_candidate_policies(
    regulation_text: str,
    policy_embeddings_dir: Path = DEFAULT_POLICY_EMBEDDINGS_DIR,
    top_k: int = DEFAULT_TOP_K,
) -> list[PolicyMatch]:
    """Return the top_k *distinct* policies most semantically similar to a
    regulation's text.

    Uses the regulation's own text as the query -- not a separately typed
    question -- since the goal is "which policies relate to what this
    regulation says," not answering a question about it.

    Retrieves more than top_k raw chunks and deduplicates to one (the
    best-scoring) chunk per policy, so a policy split into multiple chunks
    can't occupy more than one of the top_k result slots.
    """
    retriever = Retriever.from_dir(policy_embeddings_dir)
    raw_results = retriever.retrieve(regulation_text, top_k=top_k * OVERFETCH_FACTOR)

    best_per_policy: dict[str, PolicyMatch] = {}
    for r in raw_results:
        match = PolicyMatch(policy_id=r.document_id, title=r.title, score=r.score, text=r.text)
        existing = best_per_policy.get(match.policy_id)
        if existing is None or match.score > existing.score:
            best_per_policy[match.policy_id] = match

    deduped = sorted(best_per_policy.values(), key=lambda m: m.score, reverse=True)
    return deduped[:top_k]
