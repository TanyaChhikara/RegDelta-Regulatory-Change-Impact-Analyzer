"""Tests for src.embeddings.vector_store."""

import numpy as np

from src.embeddings.vector_store import VectorStore, _normalize


def test_normalize_scales_vector_to_unit_length():
    vec = np.array([3.0, 4.0])  # 3-4-5 triangle, norm = 5
    result = _normalize(vec)
    assert np.isclose(np.linalg.norm(result), 1.0)
    assert np.allclose(result, [0.6, 0.8])


def test_normalize_handles_zero_vector_without_dividing_by_zero():
    vec = np.array([0.0, 0.0])
    result = _normalize(vec)
    assert np.allclose(result, [0.0, 0.0])


def test_add_and_len():
    store = VectorStore()
    assert len(store) == 0
    store.add("a", [1.0, 0.0], {"title": "A"})
    assert len(store) == 1


def test_search_returns_most_similar_vector_first():
    store = VectorStore()
    store.add("same_direction", [1.0, 0.0], {"label": "same"})
    store.add("opposite_direction", [-1.0, 0.0], {"label": "opposite"})
    store.add("orthogonal", [0.0, 1.0], {"label": "orthogonal"})

    results = store.search([1.0, 0.0], top_k=3)

    assert results[0]["id"] == "same_direction"
    assert np.isclose(results[0]["score"], 1.0)
    assert results[-1]["id"] == "opposite_direction"
    assert np.isclose(results[-1]["score"], -1.0)


def test_search_respects_top_k():
    store = VectorStore()
    for i in range(10):
        store.add(f"id_{i}", [float(i), 1.0], {"i": i})

    results = store.search([5.0, 1.0], top_k=3)
    assert len(results) == 3


def test_search_top_k_larger_than_store_size_returns_all():
    store = VectorStore()
    store.add("only_one", [1.0, 0.0], {})
    results = store.search([1.0, 0.0], top_k=10)
    assert len(results) == 1


def test_search_on_empty_store_returns_empty_list():
    store = VectorStore()
    assert store.search([1.0, 0.0], top_k=5) == []


def test_search_result_includes_metadata():
    store = VectorStore()
    store.add(
        "doc_1", [1.0, 0.0], {"title": "Some notification", "reference_number": "RBI/2026-27/1"}
    )
    results = store.search([1.0, 0.0], top_k=1)
    assert results[0]["metadata"]["title"] == "Some notification"
    assert results[0]["metadata"]["reference_number"] == "RBI/2026-27/1"


def test_save_and_load_roundtrip_preserves_search_results(tmp_path):
    store = VectorStore()
    store.add("a", [1.0, 0.0, 0.0], {"title": "A"})
    store.add("b", [0.0, 1.0, 0.0], {"title": "B"})
    store.add("c", [0.0, 0.0, 1.0], {"title": "C"})

    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)

    assert len(loaded) == len(store)

    original_results = store.search([1.0, 0.1, 0.0], top_k=3)
    loaded_results = loaded.search([1.0, 0.1, 0.0], top_k=3)

    assert [r["id"] for r in original_results] == [r["id"] for r in loaded_results]
    for orig, loaded_r in zip(original_results, loaded_results):
        assert np.isclose(orig["score"], loaded_r["score"])


def test_save_creates_expected_files(tmp_path):
    store = VectorStore()
    store.add("a", [1.0, 0.0], {"title": "A"})
    store.save(tmp_path)

    assert (tmp_path / "vectors.npy").exists()
    assert (tmp_path / "metadata.jsonl").exists()
