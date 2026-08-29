"""
Retrieval evaluation test cases.

Each case pairs a natural-language query with the RBI reference number of
the document that should be retrieved for it. Reference numbers (not
internal chunk/document ids, which are content hashes and change across
re-fetches) are used as the stable identifier, since they're RBI's own
real, durable identifier for a document.

These cases are grounded in real documents confirmed present in the corpus
via live RBI RSS feed checks made while building M2 and M6 -- not invented
data. As the corpus grows (more fetcher runs over time, or a future
historical backfill), add more cases here rather than replacing these.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalCase:
    query: str
    # None means this is a negative case: no document in the corpus should
    # be a strong match, and we're checking whether retrieval (correctly)
    # returns a weak/low-confidence result rather than a false positive.
    expected_reference_number: str | None
    description: str = ""


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        query="interest rate ceiling for commercial bank deposits",
        expected_reference_number="RBI/2026-27/243",
        description="Commercial Banks Interest Rate on Deposits Third Amendment",
    ),
    EvalCase(
        query="cap on savings rates",
        expected_reference_number="RBI/2026-27/243",
        description="Same document as above, deliberately different wording "
        "(no literal word overlap) -- tests genuine semantic match, not keyword luck",
    ),
    EvalCase(
        query="deposit interest rate rules for small finance banks",
        expected_reference_number="RBI/2026-27/244",
        description="Small Finance Banks Interest Rate on Deposits Third Amendment",
    ),
    EvalCase(
        query="interest rate directions for urban cooperative banks",
        expected_reference_number="RBI/2026-27/247",
        description="Urban Co-operative Banks Interest Rate on Deposits Third Amendment",
    ),
    EvalCase(
        query="deposit rate amendment for regional rural banks",
        expected_reference_number="RBI/2026-27/246",
        description="Regional Rural Banks Interest Rate on Deposits Third Amendment",
    ),
    EvalCase(
        query="cash reserve ratio requirement for rural cooperative banks",
        expected_reference_number="RBI/2026-27/242",
        description="Rural Co-operative Banks CRR and SLR Fourth Amendment",
    ),
    EvalCase(
        query="statutory liquidity ratio exemption for small finance banks",
        expected_reference_number="RBI/2026-27/239",
        description="Small Finance Banks CRR and SLR Fourth Amendment",
    ),
    EvalCase(
        query="FCNR deposit relaxation extended to August 2026",
        expected_reference_number="RBI/2026-27/245",
        description="Local Area Banks amendment -- extends the FCNR(B)/NRE relaxation "
        "deadline; several sibling documents share this exact provision, so a "
        "different-but-related reference number in the top results would still "
        "be a reasonable match worth noting, even if this exact one isn't first",
    ),
    EvalCase(
        query="state government securities auction results",
        expected_reference_number=None,
        description="Should match an SDL auction press release by topic, but we "
        "haven't pinned an exact reference number for it -- reported as informational, "
        "not scored against a specific expected document",
    ),
    EvalCase(
        query="fraud prevention rules for banks",
        expected_reference_number=None,
        description="Negative case: the corpus (as of the last confirmed fetch) has no "
        "fraud-related document. Correct behavior is a low top score, not a confident "
        "match on an unrelated document -- this is the exact failure mode found "
        "manually before this eval script existed.",
    ),
]
