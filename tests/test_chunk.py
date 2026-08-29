"""Tests for src.chunking.chunk."""

from src.chunking.chunk import (
    chunk_document,
    count_tokens,
    pack_paragraphs_into_chunks,
    split_into_paragraphs,
)

SHORT_DOC = {
    "document_id": "doc_short",
    "source_feed": "notifications",
    "title": "Short Notification",
    "reference_number": "RBI/2026-27/200",
    "master_direction_refs": ["13000"],
    "pub_date": "Tue, 25 Aug 2026 18:30:00",
    "source_url": "https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=1",
    "clean_text": "RBI/2026-27/200\nA short notification body.\nChief General Manager",
}


def test_count_tokens_returns_positive_int_for_nonempty_text():
    assert count_tokens("Hello world") > 0


def test_count_tokens_returns_zero_for_empty_text():
    assert count_tokens("") == 0


def test_count_tokens_fallback_used_when_encoding_unavailable(monkeypatch):
    """When tiktoken's encoding can't be loaded (e.g. no network access to
    download it, as happens on restricted networks or air-gapped CI), token
    counting should degrade to a word-count approximation instead of
    crashing the whole pipeline.
    """
    import src.chunking.chunk as chunk_module

    monkeypatch.setattr(chunk_module, "_ENCODING", None)
    result = chunk_module.count_tokens("one two three four five")
    assert result == int(5 * chunk_module._FALLBACK_TOKENS_PER_WORD)


def test_split_into_paragraphs_drops_blank_lines():
    text = "Line one.\n\nLine two.\n   \nLine three."
    assert split_into_paragraphs(text) == ["Line one.", "Line two.", "Line three."]


def test_pack_paragraphs_into_chunks_single_chunk_when_short():
    paragraphs = ["Short paragraph one.", "Short paragraph two."]
    chunks = pack_paragraphs_into_chunks(paragraphs, target_tokens=500, overlap_tokens=50)
    assert len(chunks) == 1
    assert "Short paragraph one." in chunks[0]
    assert "Short paragraph two." in chunks[0]


def test_pack_paragraphs_into_chunks_splits_when_exceeding_target():
    # Each paragraph is deliberately long enough that a small target forces splitting.
    paragraphs = [
        f"This is paragraph number {i} with some extra words to pad it out."
        for i in range(20)
    ]
    chunks = pack_paragraphs_into_chunks(paragraphs, target_tokens=30, overlap_tokens=10)
    assert len(chunks) > 1
    # Every paragraph's content should appear somewhere in the output.
    joined = " ".join(chunks)
    for i in range(20):
        assert f"paragraph number {i} " in joined


def test_pack_paragraphs_into_chunks_overlap_carries_content_across_boundary():
    paragraphs = [f"Paragraph {i} content here padded out a fair bit more." for i in range(10)]
    chunks = pack_paragraphs_into_chunks(paragraphs, target_tokens=20, overlap_tokens=15)
    assert len(chunks) > 1
    # With overlap, the tail of chunk N should reappear at the start of chunk N+1.
    for i in range(len(chunks) - 1):
        tail_of_current = chunks[i].split("\n")[-1]
        assert tail_of_current in chunks[i + 1]


def test_chunk_document_short_doc_yields_single_chunk_with_full_text():
    chunks = chunk_document(SHORT_DOC, target_tokens=500, overlap_tokens=50)
    assert len(chunks) == 1
    assert chunks[0].text == SHORT_DOC["clean_text"]
    assert chunks[0].chunk_id == "doc_short_chunk0"
    assert chunks[0].chunk_index == 0


def test_chunk_document_carries_metadata_onto_chunks():
    chunks = chunk_document(SHORT_DOC, target_tokens=500, overlap_tokens=50)
    chunk = chunks[0]
    assert chunk.document_id == "doc_short"
    assert chunk.source_feed == "notifications"
    assert chunk.reference_number == "RBI/2026-27/200"
    assert chunk.master_direction_refs == ["13000"]


def test_chunk_document_long_doc_yields_multiple_chunks():
    long_doc = dict(SHORT_DOC)
    long_doc["document_id"] = "doc_long"
    long_doc["clean_text"] = "\n".join(
        f"This is paragraph number {i} with quite a lot of extra padding words in it."
        for i in range(30)
    )
    chunks = chunk_document(long_doc, target_tokens=30, overlap_tokens=10)
    assert len(chunks) > 1
    # chunk_index should be sequential starting from 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # every chunk id should be unique and follow the expected naming pattern
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith("doc_long_chunk") for cid in ids)


def test_chunk_document_empty_text_yields_no_chunks():
    empty_doc = dict(SHORT_DOC)
    empty_doc["clean_text"] = ""
    assert chunk_document(empty_doc) == []
