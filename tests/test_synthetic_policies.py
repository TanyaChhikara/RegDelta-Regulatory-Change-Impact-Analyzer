"""
Tests for the synthetic policy corpus (data/synthetic/policies/).

These aren't tests of application code -- they're consistency checks on
static content, to catch the manifest and the actual policy files drifting
apart (e.g. a policy renamed or removed without updating manifest.json).
"""

import json
from pathlib import Path

POLICIES_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "policies"


def _load_manifest() -> list[dict]:
    with open(POLICIES_DIR / "manifest.json", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_is_valid_json_and_nonempty():
    manifest = _load_manifest()
    assert isinstance(manifest, list)
    assert len(manifest) > 0


def test_every_manifest_entry_has_a_corresponding_file():
    manifest = _load_manifest()
    for entry in manifest:
        file_path = POLICIES_DIR / entry["file"]
        assert file_path.exists(), f"Manifest references missing file: {entry['file']}"


def test_every_manifest_entry_policy_id_and_title_appear_in_its_file():
    manifest = _load_manifest()
    for entry in manifest:
        content = (POLICIES_DIR / entry["file"]).read_text(encoding="utf-8")
        assert entry["policy_id"] in content
        assert entry["title"] in content


def test_manifest_policy_ids_are_unique():
    manifest = _load_manifest()
    ids = [entry["policy_id"] for entry in manifest]
    assert len(ids) == len(set(ids))


def test_expected_relevance_is_a_known_value():
    manifest = _load_manifest()
    for entry in manifest:
        assert entry["expected_relevance"] in {"relevant", "control"}


def test_at_least_one_relevant_and_one_control_policy_exist():
    # Sanity check on the corpus design itself: we need both kinds to be
    # useful for future gap-analysis testing.
    manifest = _load_manifest()
    relevances = {entry["expected_relevance"] for entry in manifest}
    assert "relevant" in relevances
    assert "control" in relevances


def test_no_orphan_policy_files_missing_from_manifest():
    manifest = _load_manifest()
    manifest_files = {entry["file"] for entry in manifest}
    actual_md_files = {p.name for p in POLICIES_DIR.glob("POL-*.md")}
    assert actual_md_files == manifest_files
