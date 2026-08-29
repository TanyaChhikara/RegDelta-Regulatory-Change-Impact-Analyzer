"""
Deterministic evidence verification: checks whether a gap analysis's
claimed quotes actually appear (closely, allowing minor paraphrasing) in
the source text they're attributed to.

Fulfills the "Verification Agent" role from the original project
specification without an actual agent -- see
docs/adr/006-deterministic-verification-not-agent.md for why this is a
string-matching problem, not a reasoning problem, and doesn't need an LLM
call (which could itself hallucinate about whether the first hallucination
happened).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from src.analysis.gap_analysis import GapAnalysisResult

# Below this fuzzy-match ratio, a claimed quote is considered ungrounded --
# i.e. it doesn't correspond to anything actually in the source text.
# Chosen conservatively: 0.5 tolerates real paraphrasing (an LLM
# summarizing "for the period until September 30, 2026" as "valid until
# September 30, 2026" should still pass) while still catching a claim with
# no real basis in the source.
GROUNDING_THRESHOLD = 0.5


@dataclass
class VerificationResult:
    old_requirement_grounded: bool
    old_requirement_score: float
    new_requirement_grounded: bool
    new_requirement_score: float

    @property
    def fully_verified(self) -> bool:
        return self.old_requirement_grounded and self.new_requirement_grounded


def _best_match_ratio(claim: str, source_text: str) -> float:
    """Return the highest fuzzy-match ratio between `claim` and any
    contiguous span of `source_text` the same length as `claim`.

    Uses a sliding window sized to the claim's own length rather than
    splitting into sentences first. Sentence-boundary splitting turned out
    to be unreliable on real regulatory text: a clause with embedded
    newlines and no terminal punctuation until much later gets treated as
    one long "sentence," which dilutes SequenceMatcher's ratio when
    comparing a short claim against a much longer string. A sliding window
    matched to the claim's length compares like-for-like regardless of how
    the source text happens to be punctuated.
    """
    if not claim or not source_text:
        return 0.0

    claim_normalized = " ".join(claim.split()).lower()
    source_normalized = " ".join(source_text.split()).lower()

    window_size = len(claim_normalized)
    if window_size == 0:
        return 0.0

    if len(source_normalized) <= window_size:
        return SequenceMatcher(None, claim_normalized, source_normalized).ratio()

    best = 0.0
    # Coarse stride for efficiency; fine enough given these are short
    # policy/regulation documents (a few KB), not book-length corpora.
    step = max(1, window_size // 10)
    for start in range(0, len(source_normalized) - window_size + 1, step):
        window = source_normalized[start : start + window_size]
        ratio = SequenceMatcher(None, claim_normalized, window).ratio()
        best = max(best, ratio)

    return best


def verify_gap_analysis(
    result: GapAnalysisResult, regulation_text: str, policy_text: str
) -> VerificationResult:
    """Check that a gap analysis result's claimed quotes are actually
    grounded in their claimed source texts.

    old_requirement should come from the policy; new_requirement should
    come from the regulation. If the result claims no gap (is_affected is
    False), both fields are typically null, and verification trivially
    passes -- there's nothing to ground.
    """
    if not result.is_affected:
        return VerificationResult(
            old_requirement_grounded=True,
            old_requirement_score=1.0,
            new_requirement_grounded=True,
            new_requirement_score=1.0,
        )

    old_score = _best_match_ratio(result.old_requirement or "", policy_text)
    new_score = _best_match_ratio(result.new_requirement or "", regulation_text)

    return VerificationResult(
        old_requirement_grounded=old_score >= GROUNDING_THRESHOLD,
        old_requirement_score=old_score,
        new_requirement_grounded=new_score >= GROUNDING_THRESHOLD,
        new_requirement_score=new_score,
    )
