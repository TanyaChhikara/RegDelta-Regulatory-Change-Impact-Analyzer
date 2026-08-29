"""Tests for src.analysis.policy_mapper."""

from unittest.mock import patch

from src.analysis.policy_mapper import OVERFETCH_FACTOR, PolicyMatch, find_candidate_policies
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


def test_find_candidate_policies_overfetches_using_overfetch_factor():
    with patch("src.analysis.policy_mapper.Retriever") as MockRetriever:
        mock_instance = MockRetriever.from_dir.return_value
        mock_instance.retrieve.return_value = []

        find_candidate_policies("Interest rate ceiling text.", top_k=3)

        mock_instance.retrieve.assert_called_once_with(
            "Interest rate ceiling text.", top_k=3 * OVERFETCH_FACTOR
        )


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


def test_find_candidate_policies_respects_top_k_after_dedup():
    with patch("src.analysis.policy_mapper.Retriever") as MockRetriever:
        mock_instance = MockRetriever.from_dir.return_value
        mock_instance.retrieve.return_value = [
            _make_retrieval_result("POL-001", "A", 0.9, "a"),
            _make_retrieval_result("POL-002", "B", 0.8, "b"),
            _make_retrieval_result("POL-003", "C", 0.7, "c"),
        ]

        matches = find_candidate_policies("query", top_k=2)

        assert len(matches) == 2
        assert [m.policy_id for m in matches] == ["POL-001", "POL-002"]


def test_find_candidate_policies_deduplicates_multi_chunk_document():
    """Regression test for a real finding: a policy split into multiple
    chunks (per M8 part 1) could occupy more than one of the top_k result
    slots, crowding out genuinely different candidate policies. Only the
    best-scoring chunk per policy should count toward top_k.
    """
    with patch("src.analysis.policy_mapper.Retriever") as MockRetriever:
        mock_instance = MockRetriever.from_dir.return_value
        mock_instance.retrieve.return_value = [
            _make_retrieval_result("POL-003", "NRE Policy", 0.90, "chunk 0 text"),
            _make_retrieval_result("POL-001", "Deposit Policy", 0.85, "chunk text"),
            _make_retrieval_result("POL-003", "NRE Policy", 0.80, "chunk 1 text"),
            _make_retrieval_result("POL-002", "Reserve Policy", 0.75, "chunk text"),
        ]

        matches = find_candidate_policies("query", top_k=3)

        policy_ids = [m.policy_id for m in matches]
        assert len(policy_ids) == len(set(policy_ids)), "duplicate policy in top_k results"
        assert policy_ids == ["POL-003", "POL-001", "POL-002"]
        # The higher-scoring of POL-003's two chunks (0.90) should be kept.
        pol_003_match = next(m for m in matches if m.policy_id == "POL-003")
        assert pol_003_match.score == 0.90
        assert pol_003_match.text == "chunk 0 text"


def test_find_candidate_policies_empty_store_returns_empty_list():
    with patch("src.analysis.policy_mapper.Retriever") as MockRetriever:
        mock_instance = MockRetriever.from_dir.return_value
        mock_instance.retrieve.return_value = []

        matches = find_candidate_policies("query")

        assert matches == []
