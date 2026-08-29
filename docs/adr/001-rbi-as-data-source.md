# ADR-001: Use RBI as the Primary Regulatory Data Source

**Status:** Accepted
**Date:** 2026-08-29

## Context

RegDelta requires real, publicly available regulatory data from an **Indian**
regulatory/government organization. Several Indian regulatory bodies were
considered as the primary data source, evaluated on:

1. Availability of public data
2. Quality of documentation
3. Volume of documents
4. Frequency of regulatory updates
5. Real-world usefulness
6. Suitability for RAG
7. Suitability for change detection
8. Suitability for policy-impact analysis
9. Ability to reproduce the dataset
10. Licensing/usage considerations

Candidates considered: RBI, SEBI, IRDAI, PFRDA, TRAI, MCA, IBBI.

## Decision

**Use the Reserve Bank of India (RBI) as the primary regulatory data source.**

RBI publishes:
- **Notifications** — new rules, amendments to existing rules
- **Master Directions** — consolidated, current rulebooks per topic (238 of them,
  covering 11 categories of regulated entities: commercial banks, small finance
  banks, payments banks, NBFCs, cooperative banks, etc.)
- **Circulars** — instructions issued between Master Direction updates
- **Press Releases** — announcements, policy statements

Both RSS feeds (structured XML — `notifications_rss.xml`, `pressreleases_rss.xml`)
and full-text HTML pages are publicly accessible with no API key or authentication.

## Alternatives Considered

| Source | Public data | Structured access | Volume | Update frequency | Verdict |
|---|---|---|---|---|---|
| **RBI** | Excellent | RSS feeds (XML) + HTML | ~244 Master Directions, hundreds of circulars/year | Very high (multiple/week) | **Chosen** |
| **SEBI** | Good | HTML only, no feed found in initial research | ~200 circulars/year | High | Strong second choice |
| **IRDAI** | Limited | HTML only | Lower | Monthly | Rejected — insufficient volume |
| **TRAI** | Limited | HTML only | Lower | Monthly | Rejected — not finance-relevant to the compliance domain |
| **PFRDA / MCA / IBBI** | Limited to moderate | HTML only | Lower | Lower | Rejected — narrower domain, less update velocity |

## Reasoning

1. **Structured discovery via RSS.** RBI is the only regulator among those evaluated
   with public RSS feeds for notifications and press releases. This means we don't
   have to write a fragile scraper against paginated HTML just to *discover* new
   documents — we get title, date, and link in structured XML.

2. **A once-in-a-decade natural experiment for change detection.** In late 2025, RBI
   consolidated approximately 9,445 circulars into 238 Master Directions. This gives
   us real, documented old-regulation-to-new-regulation mappings — exactly the kind
   of "what changed" signal the project needs for change detection, without us having
   to manually construct synthetic before/after pairs.

3. **Rich cross-reference structure.** RBI circulars reference Master Directions,
   which reference the RBI Act 1934, Banking Regulation Act 1949, and FEMA. This
   creates genuine multi-hop retrieval chains (circular → Master Direction → Act),
   which is the core technical challenge the whole project is built around.

4. **Domain breadth for synthetic policies.** RBI covers KYC/AML, lending norms,
   capital adequacy, digital payments, NBFC regulation, and priority sector lending —
   enough distinct domains to build a realistic, varied synthetic policy corpus
   without straining plausibility.

5. **Clean HTML over PDF.** Individual RBI circulars and Master Directions are
   published as HTML pages, not scanned PDFs. This avoids OCR-related data quality
   problems entirely for the MVP.

6. **Real-world weight.** Every scheduled bank, NBFC, and payments company in India
   must comply with RBI directions — this makes RBI the most consequential regulator
   for an Indian financial-compliance use case.

## Trade-offs / What We're Giving Up

- **No official API** (unlike the Federal Register's clean JSON API). We're relying
  on RSS + responsible HTML parsing rather than a stable, versioned interface. Page
  structure could change without notice — we accept this risk and will build the
  parser defensively (fallback selectors, logging on parse failures).
- **Not investigating SEBI in this project**, even though it would also be a strong
  choice, particularly for capital-markets-flavored compliance policies. We may
  revisit this as a stretch goal (e.g., cross-regulatory conflict detection between
  RBI and SEBI obligations) once the MVP is solid.
- **No official bulk-download or archive API**, so historical backfill (for a larger
  corpus in later phases) will require a slower, per-page fetch strategy rather than
  a single bulk export.

## Data Access Strategy

- **Discovery:** RSS feeds (`https://www.rbi.org.in/notifications_rss.xml`,
  `pressreleases_rss.xml`) for new documents; the Notification/Circular index pages
  for historical browsing by year/month.
- **Full text:** HTTP GET on individual document URLs (HTML), parsed with
  BeautifulSoup + lxml.
- **Rate limiting:** Self-imposed limit (default 2 req/sec, configurable via
  `RBI_RATE_LIMIT` in `.env`) with a descriptive `User-Agent` header, to be a
  responsible client of a public government website — not to circumvent any access
  control.
- **Licensing:** RBI regulatory publications are public-domain government documents;
  no licensing restriction on quoting/analyzing their content for a research/
  portfolio project. We will not redistribute bulk copies of RBI's website content.

## Consequences

- M2 (data fetcher) will target `rbi.org.in`'s RSS feeds and HTML circular pages
  specifically, not a generic API client.
- EDA (M2.5) will need to characterize RBI's specific HTML structure, since we have
  no schema guarantee the way an API would provide.
- The synthetic policy corpus (M7) will be organized around RBI's regulated-entity
  categories and topic areas: KYC/AML, lending, capital adequacy, digital payments,
  and NBFC regulation.
