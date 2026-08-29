"""Tests for src.retrieval.retrieve."""

from unittest.mock import patch

from src.embeddings.vector_store import VectorStore
from src.retrieval.retrieve import RetrievalResult, Retriever


def _build_test_store() -> VectorStore:
    store = VectorStore()
    store.add(
        "chunk_1",
        [1.0, 0.0, 0.0],
        {
            "document_id": "doc_1",
            "title": "Interest Rate on Deposits Directions",
            "text": "Full text about interest rate ceilings on deposits.",
            "reference_number": "RBI/2026-27/1",
            "master_direction_refs": ["13000"],
            "pub_date": "Tue, 25 Aug 2026",
            "source_url": "https://example.com/1",
        },
    )
    store.add(
        "chunk_2",
        [0.0, 1.0, 0.0],
        {
            "document_id": "doc_2",
            "title": "Fraud Risk Management Directions",
            "text": "Full text about fraud risk management for banks.",
            "reference_number": "RBI/2026-27/2",
            "master_direction_refs": [],
            "pub_date": "Wed, 26 Aug 2026",
            "source_url": "https://example.com/2",
        },
    )
    return store


def test_retrieve_embeds_query_with_is_query_true():
    store = _build_test_store()
    retriever = Retriever(store)

    with patch("src.retrieval.retrieve.embed_texts", return_value=[[1.0, 0.0, 0.0]]) as mock_embed:
        retriever.retrieve("cap on savings rates", top_k=2)

    mock_embed.assert_called_once_with(["cap on savings rates"], is_query=True)


def test_retrieve_returns_results_ranked_by_similarity():
    store = _build_test_store()
    retriever = Retriever(store)

    with patch("src.retrieval.retrieve.embed_texts", return_value=[[1.0, 0.0, 0.0]]):
        results = retriever.retrieve("interest rate query", top_k=2)

    assert len(results) == 2
    assert results[0].document_id == "doc_1"  # exact-direction match should rank first
    assert results[0].score > results[1].score


def test_retrieve_result_fields_populated_correctly():
    store = _build_test_store()
    retriever = Retriever(store)

    with patch("src.retrieval.retrieve.embed_texts", return_value=[[1.0, 0.0, 0.0]]):
        results = retriever.retrieve("query", top_k=1)

    result = results[0]
    assert isinstance(result, RetrievalResult)
    assert result.chunk_id == "chunk_1"
    assert result.title == "Interest Rate on Deposits Directions"
    assert result.reference_number == "RBI/2026-27/1"
    assert result.master_direction_refs == ["13000"]
    assert result.source_url == "https://example.com/1"


def test_retrieve_respects_top_k():
    store = _build_test_store()
    retriever = Retriever(store)

    with patch("src.retrieval.retrieve.embed_texts", return_value=[[1.0, 0.0, 0.0]]):
        results = retriever.retrieve("query", top_k=1)

    assert len(results) == 1


def test_retrieve_on_empty_store_returns_empty_list():
    retriever = Retriever(VectorStore())

    with patch("src.retrieval.retrieve.embed_texts", return_value=[[1.0, 0.0, 0.0]]):
        results = retriever.retrieve("query", top_k=5)

    assert results == []


def test_from_dir_loads_a_working_retriever(tmp_path):
    store = _build_test_store()
    store.save(tmp_path)

    retriever = Retriever.from_dir(tmp_path)

    with patch("src.retrieval.retrieve.embed_texts", return_value=[[1.0, 0.0, 0.0]]):
        results = retriever.retrieve("query", top_k=1)

    assert results[0].document_id == "doc_1"
