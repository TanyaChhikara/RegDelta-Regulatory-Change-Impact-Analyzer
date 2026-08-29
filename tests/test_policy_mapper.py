"""Tests for src.analysis.policy_mapper."""

from unittest.mock import patch

from src.analysis.policy_mapper import PolicyMatch, find_candidate_policies
from src.retrieval.retrieve import RetrievalResult


def _make_retrieval_result(
    document_id: str, title: str, score: float, text: str
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=f"{document_id}_chunk0",
        score=score,
        document_id=document_id,
        title=title,
        text=text,
        reference_number=None,
        master_direction_refs=[],
        pub_date=None,
        source_url="",
    )


def test_find_candidate_policies_uses_regulation_text_as_query():
    with patch("src.analysis.policy_mapper.Retriever") as MockRetriever:
        mock_instance = MockRetriever.from_dir.return_value
        mock_instance.retrieve.return_value = []

        find_candidate_policies("Interest rate ceiling text.", top_k=3)

        mock_instance.retrieve.assert_called_once_with("Interest rate ceiling text.", top_k=3)


def test_find_candidate_policies_maps_retrieval_results_to_policy_matches():
    with patch("src.analysis.policy_mapper.Retriever") as MockRetriever:
        mock_instance = MockRetriever.from_dir.return_value
        mock_instance.retrieve.return_value = [
            _make_retrieval_result("POL-001", "Deposit Interest Rate Policy", 0.85, "policy text"),
        ]

        matches = find_candidate_policies("some regulation text")

        assert len(matches) == 1
        assert isinstance(matches[0], PolicyMatch)
        assert matches[0].policy_id == "POL-001"
        assert matches[0].title == "Deposit Interest Rate Policy"
        assert matches[0].score == 0.85
        assert matches[0].text == "policy text"


def test_find_candidate_policies_respects_top_k():
    with patch("src.analysis.policy_mapper.Retriever") as MockRetriever:
        mock_instance = MockRetriever.from_dir.return_value
        mock_instance.retrieve.return_value = [
            _make_retrieval_result("POL-001", "A", 0.9, "a"),
            _make_retrieval_result("POL-002", "B", 0.8, "b"),
        ]

        find_candidate_policies("query", top_k=2)

        mock_instance.retrieve.assert_called_once_with("query", top_k=2)


def test_find_candidate_policies_empty_store_returns_empty_list():
    with patch("src.analysis.policy_mapper.Retriever") as MockRetriever:
        mock_instance = MockRetriever.from_dir.return_value
        mock_instance.retrieve.return_value = []

        matches = find_candidate_policies("query")

        assert matches == []
