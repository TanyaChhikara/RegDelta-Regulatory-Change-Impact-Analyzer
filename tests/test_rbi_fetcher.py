"""
Tests for src.ingestion.rbi_fetcher.

The sample HTML/text below is a real RSS <description> payload taken directly
from https://www.rbi.org.in/notifications_rss.xml (fetched 2026-08-29), not a
synthetic example -- so these tests double as a regression check against RBI's
actual feed structure.
"""

from src.ingestion.rbi_fetcher import (
    extract_clean_text,
    extract_reference_number,
    make_document_id,
    parse_entry,
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
    assert doc.document_id == make_document_id(SAMPLE_LINK, SAMPLE_TITLE)
