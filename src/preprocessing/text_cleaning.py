"""
Table-aware plain-text extraction from RBI's HTML.

BeautifulSoup's plain `get_text()` treats every table cell as just another
text node, with no awareness of rows or columns. For a document that's
mostly prose this is harmless, but RBI's auction-result and market-operations
tables are genuinely tabular (multiple columns per row), and flattening them
loses which value belongs to which column entirely -- e.g. "Notified Amount
/ Tenor / Window Timing / Date of Reversal" all become separate lines with no
indication of the pairing.

This module renders each <table> as "label: value | label: value | ..." per
row before the rest of the document is converted to plain text, so tabular
structure survives into `clean_text`.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, NavigableString, Tag


def _row_cells(row: Tag) -> list[str]:
    return [cell.get_text(separator=" ", strip=True) for cell in row.find_all(["td", "th"])]


def render_table_as_text(table: Tag) -> str:
    """Render a single <table> as readable "label: value" lines, one per row.

    Handles a real quirk seen in RBI's tables: a data row with one more cell
    than the header row, because the table has an unlabeled leading
    serial-number column (e.g. "1", "2", ...). In that case the extra
    leading cell is dropped from the pairing rather than misaligning every
    column by one.
    """
    rows = table.find_all("tr")
    if not rows:
        return table.get_text(separator=" ", strip=True)

    parsed_rows = [_row_cells(row) for row in rows]
    parsed_rows = [row for row in parsed_rows if any(cell for cell in row)]
    if not parsed_rows:
        return ""

    header: list[str] | None = None
    data_rows = parsed_rows
    if rows[0].find("th"):
        header = parsed_rows[0]
        data_rows = parsed_rows[1:]

    lines: list[str] = []
    for row_cells in data_rows:
        if not row_cells:
            continue

        aligned_header = header
        aligned_cells = row_cells

        if aligned_header and len(row_cells) == len(aligned_header) + 1:
            # Unlabeled leading serial-number column -- drop it, don't
            # misalign every other column by one.
            aligned_cells = row_cells[1:]

        if aligned_header and len(aligned_header) == len(aligned_cells):
            pairs = [f"{h}: {v}" for h, v in zip(aligned_header, aligned_cells) if v]
            lines.append(" | ".join(pairs))
        else:
            # Column counts still don't line up (e.g. no header at all, or a
            # mismatch we don't have a rule for) -- fall back to a plain
            # pipe-joined row. Still far better than full flattening: at
            # least cells from the same row stay grouped together.
            lines.append(" | ".join(v for v in row_cells if v))

    return "\n".join(lines)


def extract_clean_text_table_aware(raw_html: str) -> str:
    """Like extract_clean_text, but tables are rendered as label:value rows
    instead of being flattened into an unstructured stream of cell values.
    """
    soup = BeautifulSoup(raw_html, "lxml")

    for table in soup.find_all("table"):
        if table.parent is None:
            # Already detached because an ancestor table was replaced first
            # (nested tables are rare in RBI's HTML, but handle it safely).
            continue
        rendered = render_table_as_text(table)
        table.replace_with(NavigableString(f"\n{rendered}\n"))

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)
