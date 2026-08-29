"""
RBI circular/notification fetcher.

Fetches items from RBI's public RSS feeds (Notifications, Press Releases) and
extracts the embedded full-text HTML into clean, structured records.

Why RSS and not per-page scraping
----------------------------------
RBI's individual notification pages (NotificationUser.aspx?Id=...&Mode=0) are
ASP.NET Web Forms pages driven by server-side postbacks (__doPostBack). A plain
HTTP GET with a document ID in the query string does not render that specific
item -- it falls back to the default listing page. The RSS feed's
<description> field, however, already embeds the full text of each
notification as HTML (reference number, date, title, body, cross-references,
signature block). We treat the RSS feed as the primary full-text source, not
merely a discovery/index mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rbi_fetcher")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

RBI_FEEDS = {
    "notifications": "https://www.rbi.org.in/notifications_rss.xml",
    "press_releases": "https://www.rbi.org.in/pressreleases_rss.xml",
}

USER_AGENT = os.getenv(
    "RBI_USER_AGENT", "RegDelta/0.1 (research project; contact@example.com)"
)
RATE_LIMIT = float(os.getenv("RBI_RATE_LIMIT", "2"))  # requests per second
MIN_INTERVAL_SECONDS = 1.0 / RATE_LIMIT if RATE_LIMIT > 0 else 0.5

# Matches RBI's standard reference number format, e.g. "RBI/2026-27/248"
REFERENCE_NUMBER_PATTERN = re.compile(r"RBI/20\d{2}-\d{2}/\d+")


@dataclass
class RBIDocument:
    """A single fetched RBI notification or press release."""

    source_feed: str  # "notifications" or "press_releases"
    title: str
    reference_number: str | None
    pub_date: str | None
    source_url: str
    raw_html: str
    clean_text: str
    fetched_at: str
    document_id: str  # stable hash for dedup across repeated fetches


def _polite_get(url: str) -> requests.Response:
    """HTTP GET with a descriptive User-Agent. Raises on non-2xx responses."""
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    return response


def fetch_feed_entries(feed_key: str) -> list[dict]:
    """Fetch and parse an RBI RSS feed, returning raw feedparser entries."""
    if feed_key not in RBI_FEEDS:
        raise ValueError(f"Unknown feed '{feed_key}'. Choose from {list(RBI_FEEDS)}.")

    url = RBI_FEEDS[feed_key]
    logger.info("Fetching feed: %s", url)
    response = _polite_get(url)
    parsed = feedparser.parse(response.content)

    if parsed.bozo:
        # feedparser sets `bozo` when it had to recover from malformed XML
        # (RBI's feed has a leading byte-order-mark, for example). This is
        # expected and not fatal -- we log it and keep whatever entries were
        # still recovered.
        logger.warning("Feed parsed with recoverable issues: %s", parsed.bozo_exception)

    logger.info("Fetched %d entries from '%s'", len(parsed.entries), feed_key)
    return parsed.entries


def extract_clean_text(raw_html: str) -> str:
    """Strip HTML from an RSS <description> field, returning readable plain text."""
    soup = BeautifulSoup(raw_html, "lxml")
    text = soup.get_text(separator="\n")
    # The source HTML is a <table> of <p> tags; get_text() leaves behind a lot
    # of blank lines from that structure, so collapse them.
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def extract_reference_number(text: str) -> str | None:
    """Pull the RBI reference number (e.g. 'RBI/2026-27/248') out of the text, if present."""
    match = REFERENCE_NUMBER_PATTERN.search(text)
    return match.group(0) if match else None


def make_document_id(source_url: str, title: str) -> str:
    """Stable identifier for dedup across repeated fetches of the same feed."""
    key = f"{source_url}|{title}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def parse_entry(entry: dict, feed_key: str) -> RBIDocument:
    """Convert one feedparser entry into a structured RBIDocument."""
    raw_html = entry.get("summary", "") or entry.get("description", "")
    clean_text = extract_clean_text(raw_html)
    title = entry.get("title", "").strip()
    source_url = entry.get("link", "")

    return RBIDocument(
        source_feed=feed_key,
        title=title,
        reference_number=extract_reference_number(clean_text),
        pub_date=entry.get("published"),
        source_url=source_url,
        raw_html=raw_html,
        clean_text=clean_text,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        document_id=make_document_id(source_url, title),
    )


def fetch_documents(feed_key: str) -> list[RBIDocument]:
    """Fetch a feed and return it as a list of structured documents."""
    entries = fetch_feed_entries(feed_key)
    return [parse_entry(entry, feed_key) for entry in entries]


def save_documents(documents: list[RBIDocument], output_dir: Path) -> Path:
    """Write documents as JSON Lines (one document per line) to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"rbi_fetch_{timestamp}.jsonl"

    with output_path.open("w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

    logger.info("Saved %d documents to %s", len(documents), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch RBI regulatory documents from RSS feeds.")
    parser.add_argument(
        "--feed",
        choices=[*RBI_FEEDS.keys(), "all"],
        default="notifications",
        help="Which RBI feed to fetch.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory to write fetched JSONL files.",
    )
    args = parser.parse_args()

    feed_keys = list(RBI_FEEDS) if args.feed == "all" else [args.feed]
    output_dir = Path(args.output_dir)

    for feed_key in feed_keys:
        documents = fetch_documents(feed_key)
        save_documents(documents, output_dir)
        time.sleep(MIN_INTERVAL_SECONDS)  # be polite between feeds


if __name__ == "__main__":
    main()
