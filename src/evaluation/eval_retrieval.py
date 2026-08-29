"""
Retrieval evaluation: measures Recall@K and Mean Reciprocal Rank (MRR)
against a small, real test set (see eval_cases.py), and specifically flags
the "confident-looking but irrelevant" failure mode -- a negative-case query
returning a high similarity score despite no genuinely relevant document
existing in the corpus.

This is intentionally small (10 cases) rather than the full 100-case
benchmark from the project's original roadmap. The goal right now is a real,
grounded first measurement -- Baseline 2 in the project's staged baseline
comparison -- not exhaustive coverage. Grow this test set as the corpus grows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.eval_cases import EVAL_CASES, EvalCase
from src.retrieval.retrieve import RetrievalResult, Retriever

# A true-negative-case top score at or above this is flagged as a possible
# false-positive risk -- confident-looking but likely irrelevant.
#
# This is not an enforced filtering threshold anywhere in retrieve.py, and
# 0.6 is still a rough starting point, not a validated cutoff -- it's based
# on exactly one clean true-negative data point so far (a real run scored
# a genuine false positive at 0.615, while a genuine correct match on a
# different query scored 0.783). That gap is a good sign the score does
# carry real signal, but one data point on the "wrong" side isn't enough to
# lock in a specific number. Revisit as more true-negative cases accumulate.
FALSE_POSITIVE_RISK_SCORE = 0.6


@dataclass
class CaseResult:
    case: EvalCase
    results: list[RetrievalResult]
    rank: int | None  # 1-indexed rank of the expected document, if found
    reciprocal_rank: float | None  # None for cases with no expected document
    top_score: float | None


def evaluate_case(retriever: Retriever, case: EvalCase, top_k: int) -> CaseResult:
    results = retriever.retrieve(case.query, top_k=top_k)
    top_score = results[0].score if results else None

    if case.expected_reference_number is None:
        return CaseResult(
            case=case, results=results, rank=None, reciprocal_rank=None, top_score=top_score
        )

    rank = None
    for i, result in enumerate(results, start=1):
        if result.reference_number == case.expected_reference_number:
            rank = i
            break

    reciprocal_rank = 1.0 / rank if rank else 0.0
    return CaseResult(
        case=case, results=results, rank=rank, reciprocal_rank=reciprocal_rank, top_score=top_score
    )


def run_evaluation(retriever: Retriever, cases: list[EvalCase], top_k: int = 5) -> dict:
    case_results = [evaluate_case(retriever, case, top_k) for case in cases]

    scored = [r for r in case_results if r.case.expected_reference_number is not None]
    recall_at_k = sum(1 for r in scored if r.rank is not None) / len(scored) if scored else None
    mrr = sum(r.reciprocal_rank for r in scored) / len(scored) if scored else None

    return {"case_results": case_results, "recall_at_k": recall_at_k, "mrr": mrr, "top_k": top_k}


def print_report(report: dict) -> None:
    top_k = report["top_k"]
    print(f"\n{'=' * 80}\nRETRIEVAL EVALUATION (top_k={top_k})\n{'=' * 80}")

    for i, cr in enumerate(report["case_results"], start=1):
        print(f"\n[{i}] Query: {cr.case.query!r}")
        if cr.case.description:
            print(f"    {cr.case.description}")

        if cr.case.expected_reference_number is None:
            flag = ""
            if (
                cr.case.is_true_negative
                and cr.top_score is not None
                and cr.top_score >= FALSE_POSITIVE_RISK_SCORE
            ):
                flag = "  <-- possible false-positive risk (high score, no expected match)"
            label = "Negative case" if cr.case.is_true_negative else "Informational (unscored)"
            top_title = cr.results[0].title if cr.results else "(no results)"
            score_str = f"{cr.top_score:.3f}" if cr.top_score is not None else "N/A"
            print(f"    {label}. Top result: {top_title}  score={score_str}{flag}")
        else:
            status = f"FOUND at rank {cr.rank}" if cr.rank else "NOT FOUND"
            print(f"    Expected: {cr.case.expected_reference_number}  ->  {status}")

    print(f"\n{'-' * 80}")
    recall_str = f"{report['recall_at_k']:.0%}" if report["recall_at_k"] is not None else "N/A"
    mrr_str = f"{report['mrr']:.3f}" if report["mrr"] is not None else "N/A"
    print(f"Recall@{top_k}: {recall_str}")
    print(f"MRR@{top_k}:    {mrr_str}")
    print(f"{'-' * 80}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality against known test cases."
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results per query.")
    parser.add_argument(
        "--embeddings-dir", default="data/embeddings", help="Directory holding the vector store."
    )
    args = parser.parse_args()

    retriever = Retriever.from_dir(Path(args.embeddings_dir))
    report = run_evaluation(retriever, EVAL_CASES, top_k=args.top_k)
    print_report(report)


if __name__ == "__main__":
    main()
