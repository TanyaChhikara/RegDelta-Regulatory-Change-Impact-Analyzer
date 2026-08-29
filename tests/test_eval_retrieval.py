"""Tests for src.evaluation.eval_retrieval."""

from unittest.mock import MagicMock

from src.evaluation.eval_cases import EvalCase
from src.evaluation.eval_retrieval import evaluate_case, print_report, run_evaluation
from src.retrieval.retrieve import RetrievalResult


def _make_result(reference_number: str | None, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="c", score=score, document_id="d", title="T", text="text",
        reference_number=reference_number, master_direction_refs=[], pub_date=None, source_url="",
    )


def test_evaluate_case_found_at_rank_1():
    case = EvalCase(query="q", expected_reference_number="RBI/1")
    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_result("RBI/1", 0.9), _make_result("RBI/2", 0.5)]

    result = evaluate_case(retriever, case, top_k=5)

    assert result.rank == 1
    assert result.reciprocal_rank == 1.0
    assert result.top_score == 0.9


def test_evaluate_case_found_at_rank_3():
    case = EvalCase(query="q", expected_reference_number="RBI/3")
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        _make_result("RBI/1", 0.9), _make_result("RBI/2", 0.8), _make_result("RBI/3", 0.7)
    ]

    result = evaluate_case(retriever, case, top_k=5)

    assert result.rank == 3
    assert result.reciprocal_rank == 1.0 / 3


def test_evaluate_case_not_found_gives_none_rank_and_zero_reciprocal_rank():
    case = EvalCase(query="q", expected_reference_number="RBI/999")
    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_result("RBI/1", 0.9), _make_result("RBI/2", 0.5)]

    result = evaluate_case(retriever, case, top_k=5)

    assert result.rank is None
    assert result.reciprocal_rank == 0.0


def test_evaluate_case_negative_case_reports_top_score_without_rank():
    case = EvalCase(query="q", expected_reference_number=None, is_true_negative=True)
    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_result("RBI/1", 0.65)]

    result = evaluate_case(retriever, case, top_k=5)

    assert result.rank is None
    assert result.reciprocal_rank is None
    assert result.top_score == 0.65


def test_evaluate_case_informational_case_also_reports_top_score_without_rank():
    """An informational case (a relevant document likely exists, but there's
    no stable reference number to score against) is structurally identical
    to a true negative case at the evaluate_case level -- the distinction
    only matters for reporting/flagging, tested separately in
    test_print_report behavior via the is_true_negative field itself.
    """
    case = EvalCase(query="q", expected_reference_number=None, is_true_negative=False)
    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_result("RBI/1", 0.78)]

    result = evaluate_case(retriever, case, top_k=5)

    assert result.rank is None
    assert result.reciprocal_rank is None
    assert result.top_score == 0.78


def test_evaluate_case_empty_results_handled_gracefully():
    case = EvalCase(query="q", expected_reference_number="RBI/1")
    retriever = MagicMock()
    retriever.retrieve.return_value = []

    result = evaluate_case(retriever, case, top_k=5)

    assert result.rank is None
    assert result.reciprocal_rank == 0.0
    assert result.top_score is None


def test_run_evaluation_computes_recall_and_mrr_correctly():
    retriever = MagicMock()

    def fake_retrieve(query, top_k):
        if query == "found_first":
            return [_make_result("A", 0.9)]
        if query == "found_third":
            return [_make_result("X", 0.9), _make_result("Y", 0.8), _make_result("B", 0.7)]
        if query == "not_found":
            return [_make_result("Z", 0.9)]
        return []

    retriever.retrieve.side_effect = fake_retrieve

    cases = [
        EvalCase(query="found_first", expected_reference_number="A"),
        EvalCase(query="found_third", expected_reference_number="B"),
        EvalCase(query="not_found", expected_reference_number="C"),
    ]

    report = run_evaluation(retriever, cases, top_k=5)

    # recall: 2 of 3 found (found_first, found_third) -> 2/3
    assert abs(report["recall_at_k"] - (2 / 3)) < 1e-9
    # mrr: (1/1 + 1/3 + 0) / 3
    expected_mrr = (1.0 + (1.0 / 3) + 0.0) / 3
    assert abs(report["mrr"] - expected_mrr) < 1e-9


def test_run_evaluation_ignores_negative_cases_in_aggregate_metrics():
    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_result("A", 0.9)]

    cases = [
        EvalCase(query="q1", expected_reference_number="A"),  # found, rank 1
        EvalCase(query="q2", expected_reference_number=None),  # negative case, excluded
    ]

    report = run_evaluation(retriever, cases, top_k=5)

    # Only the one positive case should count toward recall/MRR.
    assert report["recall_at_k"] == 1.0
    assert report["mrr"] == 1.0


def test_run_evaluation_handles_empty_case_list():
    retriever = MagicMock()
    report = run_evaluation(retriever, [], top_k=5)

    assert report["recall_at_k"] is None
    assert report["mrr"] is None


def test_print_report_flags_true_negative_with_high_score(capsys):
    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_result("RBI/1", 0.9)]
    case = EvalCase(query="fraud query", expected_reference_number=None, is_true_negative=True)

    report = run_evaluation(retriever, [case], top_k=5)
    print_report(report)

    captured = capsys.readouterr()
    assert "possible false-positive risk" in captured.out
    assert "Negative case" in captured.out


def test_print_report_does_not_flag_informational_case_with_high_score(capsys):
    """Regression test: an informational case (real match exists, no stable
    id to score against) must NOT be flagged as a false-positive risk just
    because its score is high -- a high score there means retrieval worked
    correctly, the opposite of a true negative case's high score.
    """
    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_result("RBI/1", 0.9)]
    case = EvalCase(query="auction query", expected_reference_number=None, is_true_negative=False)

    report = run_evaluation(retriever, [case], top_k=5)
    print_report(report)

    captured = capsys.readouterr()
    assert "possible false-positive risk" not in captured.out
    assert "Informational (unscored)" in captured.out
