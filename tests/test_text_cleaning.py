"""Tests for src.preprocessing.text_cleaning."""

from bs4 import BeautifulSoup

from src.preprocessing.text_cleaning import extract_clean_text_table_aware, render_table_as_text


def test_simple_table_with_header_pairs_correctly():
    html = """
    <table>
      <tr><th>Notified Amount</th><th>Tenor</th></tr>
      <tr><td>6,00,000</td><td>15</td></tr>
    </table>
    """
    result = extract_clean_text_table_aware(html)
    assert "Notified Amount: 6,00,000" in result
    assert "Tenor: 15" in result


def test_table_with_unlabeled_leading_serial_column_drops_it_correctly():
    """Regression test for a real reported issue: a data row with one extra
    leading cell (an unlabeled serial number) must not shift every other
    column's pairing by one position.
    """
    html = """
    <table>
      <tr>
        <th>Notified Amount (\u20b9 crore)</th>
        <th>Tenor (day)</th>
        <th>Window Timing</th>
        <th>Date of Reversal</th>
      </tr>
      <tr>
        <td>1</td><td>6,00,000</td><td>15</td>
        <td>09:30 AM to 10:00 AM</td>
        <td>September 15, 2026 (Tuesday)</td>
      </tr>
    </table>
    """
    result = extract_clean_text_table_aware(html)
    assert "Notified Amount (\u20b9 crore): 6,00,000" in result
    assert "Tenor (day): 15" in result
    assert "Window Timing: 09:30 AM to 10:00 AM" in result
    assert "Date of Reversal: September 15, 2026 (Tuesday)" in result
    # The stray serial number "1" must not have been paired with the first
    # header, which would silently corrupt every value in the row.
    assert "Notified Amount (\u20b9 crore): 1" not in result


def test_table_without_header_falls_back_to_pipe_joined_cells():
    html = """
    <table>
      <tr><td>I.</td><td>Notified Amount</td><td>\u20b934,000 crore</td></tr>
      <tr><td>II.</td><td>Cut off Price</td><td>100.17</td></tr>
    </table>
    """
    result = extract_clean_text_table_aware(html)
    assert "I. | Notified Amount | \u20b934,000 crore" in result
    assert "II. | Cut off Price | 100.17" in result


def test_plain_document_with_no_tables_unaffected():
    html = "<p>RBI/2026-27/248</p><p>Some regulatory text here.</p>"
    result = extract_clean_text_table_aware(html)
    assert result == "RBI/2026-27/248\nSome regulatory text here."


def test_table_preserves_position_relative_to_surrounding_paragraphs():
    html = """
    <p>Before the table.</p>
    <table><tr><th>A</th></tr><tr><td>1</td></tr></table>
    <p>After the table.</p>
    """
    result = extract_clean_text_table_aware(html)
    lines = result.splitlines()
    assert lines[0] == "Before the table."
    assert lines[-1] == "After the table."
    assert any("A: 1" in line for line in lines)


def test_render_table_as_text_handles_table_with_no_rows():
    html = "<table></table>"
    table = BeautifulSoup(html, "lxml").find("table")
    assert render_table_as_text(table) == ""
