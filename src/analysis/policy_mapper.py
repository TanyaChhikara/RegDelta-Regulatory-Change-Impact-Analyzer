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
    """Return the top_k policies most semantically similar to a regulation's text.

    Uses the regulation's own text as the query -- not a separately typed
    question -- since the goal is "which policies relate to what this
    regulation says," not answering a question about it.
    """
    retriever = Retriever.from_dir(policy_embeddings_dir)
    results = retriever.retrieve(regulation_text, top_k=top_k)

    return [
        PolicyMatch(policy_id=r.document_id, title=r.title, score=r.score, text=r.text)
        for r in results
    ]
