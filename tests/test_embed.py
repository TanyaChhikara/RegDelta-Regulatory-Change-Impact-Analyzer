"""Tests for src.embeddings.embed."""

import json
from unittest.mock import MagicMock, patch

from src.embeddings.embed import (
    FAKE_EMBEDDING_DIM,
    _fake_embed_one,
    embed_chunks,
    embed_texts,
)


def test_fake_embed_one_is_deterministic():
    vec1 = _fake_embed_one("hello world")
    vec2 = _fake_embed_one("hello world")
    assert vec1 == vec2


def test_fake_embed_one_differs_for_different_text():
    vec1 = _fake_embed_one("hello world")
    vec2 = _fake_embed_one("goodbye world")
    assert vec1 != vec2


def test_fake_embed_one_has_expected_dimension():
    vec = _fake_embed_one("some text")
    assert len(vec) == FAKE_EMBEDDING_DIM


def test_embed_texts_empty_list_returns_empty_list():
    assert embed_texts([]) == []


def test_embed_texts_fake_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    result = embed_texts(["text one", "text two"])
    assert len(result) == 2
    assert len(result[0]) == FAKE_EMBEDDING_DIM
    # Same text embedded twice should give the same vector (fake provider is deterministic).
    assert embed_texts(["text one"])[0] == result[0]


def test_embed_texts_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "not_a_real_provider")
    try:
        embed_texts(["some text"])
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Unknown EMBEDDING_PROVIDER" in str(e)


def test_embed_texts_openai_provider_calls_api_and_respects_response_index(monkeypatch):
    """Mocks the OpenAI client entirely -- no real API call, no API key needed.
    Also verifies embeddings are reordered by the response's own `index`
    field rather than trusted to arrive in request order.
    """
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    mock_item_0 = MagicMock(index=0, embedding=[0.1, 0.2])
    mock_item_1 = MagicMock(index=1, embedding=[0.3, 0.4])
    # Deliberately return them out of order to test the sort-by-index logic.
    mock_response = MagicMock(data=[mock_item_1, mock_item_0])

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("openai.OpenAI", return_value=mock_client):
        result = embed_texts(["first text", "second text"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    mock_client.embeddings.create.assert_called_once_with(
        input=["first text", "second text"], model="text-embedding-3-small"
    )


def test_embed_chunks_builds_store_with_correct_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")

    chunks_path = tmp_path / "chunks.jsonl"
    chunk_record = {
        "chunk_id": "doc1_chunk0",
        "document_id": "doc1",
        "chunk_index": 0,
        "text": "Some regulatory text.",
        "token_count": 4,
        "source_feed": "notifications",
        "title": "A Notification",
        "reference_number": "RBI/2026-27/1",
        "master_direction_refs": ["13000"],
        "pub_date": "Tue, 25 Aug 2026",
        "source_url": "https://example.com",
    }
    with chunks_path.open("w") as f:
        f.write(json.dumps(chunk_record) + "\n")

    store = embed_chunks(chunks_path)

    assert len(store) == 1
    assert store.ids[0] == "doc1_chunk0"
    assert store.metadata[0]["title"] == "A Notification"
    assert store.metadata[0]["reference_number"] == "RBI/2026-27/1"
    assert store.metadata[0]["master_direction_refs"] == ["13000"]


def test_embed_chunks_empty_file_returns_empty_store(tmp_path, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    chunks_path = tmp_path / "empty.jsonl"
    chunks_path.write_text("")

    store = embed_chunks(chunks_path)
    assert len(store) == 0
