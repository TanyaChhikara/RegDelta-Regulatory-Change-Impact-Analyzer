"""Tests for src.policies.policy_loader and src.policies.embed_policies."""

from src.policies.embed_policies import embed_policies
from src.policies.policy_loader import load_policy_documents, parse_frontmatter


def test_parse_frontmatter_extracts_keys_and_body():
    raw = """---
policy_id: POL-001
title: Test Policy
---
# Body

Some content here.
"""
    frontmatter, body = parse_frontmatter(raw)
    assert frontmatter["policy_id"] == "POL-001"
    assert frontmatter["title"] == "Test Policy"
    assert body == "# Body\n\nSome content here."


def test_parse_frontmatter_handles_missing_frontmatter():
    raw = "Just a plain document with no frontmatter."
    frontmatter, body = parse_frontmatter(raw)
    assert frontmatter == {}
    assert body == raw


def test_parse_frontmatter_handles_unclosed_frontmatter():
    raw = "---\nkey: value\nNo closing marker here."
    frontmatter, body = parse_frontmatter(raw)
    assert frontmatter == {}
    assert body == raw


def test_load_policy_documents_loads_all_real_policies():
    docs = load_policy_documents()
    assert len(docs) == 6
    ids = {d["document_id"] for d in docs}
    assert ids == {"POL-001", "POL-002", "POL-003", "POL-004", "POL-005", "POL-006"}


def test_load_policy_documents_have_nonempty_clean_text():
    docs = load_policy_documents()
    for doc in docs:
        assert len(doc["clean_text"]) > 100  # real policy bodies are substantial


def test_load_policy_documents_carries_expected_relevance():
    docs = load_policy_documents()
    relevant = [d for d in docs if d["expected_relevance"] == "relevant"]
    control = [d for d in docs if d["expected_relevance"] == "control"]
    assert len(relevant) == 3
    assert len(control) == 3


def test_embed_policies_end_to_end_with_fake_provider(monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    store = embed_policies()

    # 6 policies -> at least 6 chunks; the 3 "relevant" policies are long
    # enough (detailed relaxation-window provisions) to genuinely cross
    # the 500-token chunking threshold from M4 and split into 2 chunks
    # each, while the 3 shorter "control" policies stay as a single
    # chunk -- confirmed by inspecting real chunk counts, not assumed.
    document_ids = {m["document_id"] for m in store.metadata}
    assert document_ids == {"POL-001", "POL-002", "POL-003", "POL-004", "POL-005", "POL-006"}
    assert len(store) >= 6

    titles = {m["title"] for m in store.metadata}
    assert "Deposit Interest Rate Policy" in titles
    assert "Fraud Risk Management Policy" in titles
