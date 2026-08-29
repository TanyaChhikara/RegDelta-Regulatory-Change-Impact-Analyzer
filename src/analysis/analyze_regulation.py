"""
End-to-end regulatory impact analysis: given an RBI reference number, find
its text, retrieve candidate affected policies, run gap analysis on each,
and print an evidence-first report.

This is the first point in the project where all the pieces (M2-M8) connect
into the actual thing the project is named for -- not search, not
similarity scores, but "this regulatory change affects that policy, here's
specifically why, and here's what to review."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.gap_analysis import analyze_gap
from src.analysis.policy_mapper import find_candidate_policies


def load_regulation_by_reference(
    reference_number: str, processed_path: Path = Path("data/processed/processed_documents.jsonl")
) -> dict | None:
    """Find a regulation's full record by its RBI reference number."""
    with open(processed_path, encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            if doc.get("reference_number") == reference_number:
                return doc
    return None


def print_impact_report(regulation: dict, candidate_score_threshold: float, top_k: int) -> None:
    print(f"\n{'=' * 80}")
    print(f"REGULATORY IMPACT ANALYSIS: {regulation['reference_number']}")
    print(f"{regulation['title']}")
    print(f"{'=' * 80}")

    matches = find_candidate_policies(regulation["clean_text"], top_k=top_k)

    if not matches:
        print("\nNo candidate policies found in the policy vector store.")
        return

    for match in matches:
        print(f"\n{'-' * 80}")
        print(f"Candidate policy: {match.title} ({match.policy_id})  similarity={match.score:.3f}")

        if match.score < candidate_score_threshold:
            print(
                f"  Below candidate_score_threshold ({candidate_score_threshold}) -- "
                "skipping LLM gap analysis for this candidate."
            )
            continue

        result = analyze_gap(regulation["clean_text"], match.text)

        print(f"  Affected: {result.is_affected}  (confidence: {result.confidence})")
        print(f"  Reasoning: {result.reasoning}")
        if result.is_affected:
            print(f"  Old requirement (policy): {result.old_requirement}")
            print(f"  New requirement (regulation): {result.new_requirement}")
            print(f"  Recommended action: {result.recommended_action}")

    print(f"\n{'=' * 80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a regulation's impact on the synthetic policy corpus."
    )
    parser.add_argument(
        "--reference", required=True, help="RBI reference number, e.g. RBI/2026-27/243"
    )
    parser.add_argument(
        "--processed-path",
        default="data/processed/processed_documents.jsonl",
        help="Path to processed regulation documents.",
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of candidate policies to consider."
    )
    parser.add_argument(
        "--candidate-score-threshold",
        type=float,
        default=0.0,
        help="Skip LLM gap analysis for candidates below this similarity score. "
        "Default 0.0 (analyze all top_k candidates) since no validated threshold "
        "exists yet -- see src/evaluation/eval_retrieval.py's FALSE_POSITIVE_RISK_SCORE "
        "discussion for why this isn't defaulted to a nonzero value yet.",
    )
    args = parser.parse_args()

    regulation = load_regulation_by_reference(args.reference, Path(args.processed_path))
    if regulation is None:
        print(
            f"No regulation found with reference number {args.reference!r} "
            f"in {args.processed_path}"
        )
        return

    print_impact_report(regulation, args.candidate_score_threshold, args.top_k)


if __name__ == "__main__":
    main()
