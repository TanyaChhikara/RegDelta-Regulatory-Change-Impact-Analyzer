"""
Tests for src.ingestion.rbi_fetcher.

The sample HTML/text below is a real RSS <description> payload taken directly
from https://www.rbi.org.in/notifications_rss.xml (fetched 2026-08-29), not a
synthetic example -- so these tests double as a regression check against RBI's
actual feed structure.
"""

from src.ingestion.rbi_fetcher import (
    extract_clean_text,
    extract_master_direction_refs,
    extract_reference_number,
    make_document_id,
    parse_entry,
    save_documents,
)

# A real <description> payload from the RBI notifications RSS feed.
SAMPLE_RAW_HTML = """<table width="100%" border="0" align="center" class="td">
  <tr>
    <td><p>RBI/2026-27/248<br>
    DOR.SOG(SPE).REC.214/13.03.00/2026-27</p>
    <p align="right">August 25, 2026</p>
    <p class="head" align="center">Reserve Bank of India (Rural Co-operative Banks
    &ndash; Interest Rate on Deposits) Second Amendment Directions, 2026</p>
    <p>Please refer to the
    <a href="https://www.rbi.org.in/scripts/BS_ViewMasDirections.aspx?id=13003"
    target="_blank" class="links">Reserve Bank of India (Rural Co-operative Banks
    &ndash; Interest Rate on Deposits) Directions, 2025</a>, dated November 28, 2025.</p>
    <p>2. Accordingly, in exercise of the powers conferred by Section 35A read with
    Section 56 of the Banking Regulation Act,1949, the RBI hereby issues the
    Amendment Directions hereinafter specified.</p>
    <p>5. These Amendment Directions shall modify the
    <a href="https://www.rbi.org.in/scripts/BS_ViewMasDirections.aspx?id=13003"
    target="_blank" class="links">Directions</a> as under:</p>
    <p>(Dr. Sudarsana Sahoo)<br>
    Chief General Manager</p></td>
  </tr>
</table>"""

SAMPLE_TITLE = (
    "Reserve Bank of India (Rural Co-operative Banks \u2013 Interest Rate on "
    "Deposits) Second Amendment Directions, 2026"
)
SAMPLE_LINK = "https://www.rbi.org.in/scripts/NotificationUser.aspx?Id=13690&Mode=0"


def test_extract_clean_text_strips_html_and_collapses_blank_lines():
    text = extract_clean_text(SAMPLE_RAW_HTML)

    assert "<p>" not in text
    assert "<table" not in text
    assert "RBI/2026-27/248" in text
    assert "Chief General Manager" in text
    # No blank lines should survive the collapse.
    assert "" not in text.splitlines()


def test_extract_clean_text_preserves_cross_reference_context():
    # The cross-reference to the Master Direction is exactly the kind of
    # multi-hop signal the retrieval pipeline will depend on later -- make
    # sure it survives HTML stripping as plain text, not just as a stray href.
    text = extract_clean_text(SAMPLE_RAW_HTML)
    assert "Rural Co-operative Banks" in text
    assert "Directions, 2025" in text


def test_extract_reference_number_finds_valid_rbi_reference():
    text = extract_clean_text(SAMPLE_RAW_HTML)
    assert extract_reference_number(text) == "RBI/2026-27/248"


def test_extract_reference_number_returns_none_when_absent():
    assert extract_reference_number("No reference number in this text.") is None


def test_make_document_id_is_stable_for_same_input():
    id_1 = make_document_id(SAMPLE_LINK, SAMPLE_TITLE)
    id_2 = make_document_id(SAMPLE_LINK, SAMPLE_TITLE)
    assert id_1 == id_2
    assert len(id_1) == 16


def test_make_document_id_differs_for_different_input():
    id_1 = make_document_id(SAMPLE_LINK, SAMPLE_TITLE)
    id_2 = make_document_id(SAMPLE_LINK, "A completely different title")
    assert id_1 != id_2


def test_extract_master_direction_refs_finds_linked_id():
    refs = extract_master_direction_refs(SAMPLE_RAW_HTML)
    assert refs == ["13003"]


def test_extract_master_direction_refs_deduplicates_repeated_links():
    # SAMPLE_RAW_HTML links id=13003 twice (once in the opening paragraph,
    # once in the "these Directions modify..." paragraph) -- exactly the
    # real-world pattern seen in live RBI notifications.
    refs = extract_master_direction_refs(SAMPLE_RAW_HTML)
    assert len(refs) == 1


def test_extract_master_direction_refs_returns_empty_list_when_absent():
    assert extract_master_direction_refs("<p>No links here.</p>") == []


def test_parse_entry_builds_expected_document():
    entry = {
        "title": SAMPLE_TITLE,
        "summary": SAMPLE_RAW_HTML,
        "link": SAMPLE_LINK,
        "published": "Tue, 25 Aug 2026 18:30:00",
    }

    doc = parse_entry(entry, feed_key="notifications")

    assert doc.source_feed == "notifications"
    assert doc.title == SAMPLE_TITLE
    assert doc.reference_number == "RBI/2026-27/248"
    assert doc.source_url == SAMPLE_LINK
    assert doc.pub_date == "Tue, 25 Aug 2026 18:30:00"
    assert "Chief General Manager" in doc.clean_text
    assert doc.master_direction_refs == ["13003"]
    assert doc.document_id == make_document_id(SAMPLE_LINK, SAMPLE_TITLE)


def test_save_documents_does_not_collide_across_feeds_in_same_second(tmp_path):
    """Regression test for a real bug: fetching notifications then
    press_releases back-to-back can complete within the same wall-clock
    second. Before this fix, save_documents() used a second-precision
    timestamp as the *only* uniqueness key in its filename, so the second
    feed's write silently overwrote the first feed's file -- losing an
    entire feed's worth of data with no error raised.
    """
    entry = {
        "title": SAMPLE_TITLE,
        "summary": SAMPLE_RAW_HTML,
        "link": SAMPLE_LINK,
        "published": "Tue, 25 Aug 2026 18:30:00",
    }
    notification_doc = parse_entry(entry, feed_key="notifications")
    press_release_doc = parse_entry(entry, feed_key="press_releases")

    # Simulate the real call pattern: two feeds saved in immediate succession.
    path_1 = save_documents([notification_doc], tmp_path, "notifications")
    path_2 = save_documents([press_release_doc], tmp_path, "press_releases")

    assert path_1 != path_2
    assert path_1.exists()
    assert path_2.exists()

    # Both files' content must be independently readable -- neither
    # should have been clobbered by the other.
    assert "notifications" in path_1.read_text()
    assert "press_releases" in path_2.read_text()
