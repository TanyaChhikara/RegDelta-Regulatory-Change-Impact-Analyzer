# ADR-004: Switch Default Embedding Provider to Gemini

**Status:** Accepted
**Date:** 2026-08-30

## Context

ADR-003 chose OpenAI `text-embedding-3-small` as the default embedding
provider. In practice, setting up OpenAI billing hit a real-world payment
obstacle: multiple cards were declined by the issuing bank when charging
OpenAI (a US merchant), consistent with a common pattern where Indian banks
disable international/foreign-currency transactions by default. This
blocked actually running the embedding pipeline, independent of anything in
the code.

Separately, while researching a workaround, Groq was confirmed (via Groq's
own official API reference) to **not** offer an embeddings endpoint at all --
some third-party pages suggested otherwise, but the primary source doesn't
list one, so Groq was ruled out as an alternative.

Google's Gemini API offers `gemini-embedding-001`, with a genuinely free
tier (no credit card required to obtain an API key or make embedding calls,
as long as billing isn't separately linked to the same Google Cloud
project).

## Decision

**Switch the default `EMBEDDING_PROVIDER` to `gemini`, using
`gemini-embedding-001`.** OpenAI support (built and tested in M5) remains in
the codebase as an alternative, unchanged.

## Reasoning

1. **Removes a real access barrier.** No credit card requirement means the
   pipeline can actually run without depending on a specific bank's
   international-transaction policy or issuer approval.
2. **A genuine quality feature, not just a workaround.** Unlike OpenAI's
   symmetric embeddings, `gemini-embedding-001` supports task-type-aware
   embeddings (`RETRIEVAL_DOCUMENT` for indexed text, `RETRIEVAL_QUERY` for
   search queries) -- asymmetric embeddings are a known improvement for
   retrieval specifically, since a query and the document that answers it
   are often phrased very differently. This is implemented as an `is_query`
   parameter on `embed_texts()`.
3. **No architecture cost.** The existing `EMBEDDING_PROVIDER` abstraction
   from ADR-003 was built exactly for this kind of swap -- adding a second
   real provider validates that design decision rather than requiring
   changes to it.

## Alternatives Considered

| Option | Verdict |
|---|---|
| Keep troubleshooting OpenAI billing | Deferred indefinitely -- outside this project's control (bank-side) |
| Groq | Rejected -- confirmed no embeddings endpoint via Groq's official API reference |
| Gemini (`gemini-embedding-001`) | **Chosen** -- free tier with no card required, task-type-aware embeddings |

## Consequences

- `.env.example` updated: `EMBEDDING_PROVIDER` default changed to `gemini`,
  with `GEMINI_API_KEY` and `GEMINI_EMBEDDING_MODEL` added.
- `embed_texts()` gains an `is_query: bool` parameter so documents are
  embedded with `RETRIEVAL_DOCUMENT` and search queries with
  `RETRIEVAL_QUERY`. This parameter is a no-op for the `openai` and `fake`
  providers, which have no task-type concept.
- OpenAI's embeddings and Gemini's embeddings are **not compatible** with
  each other (different vector spaces, different dimensions by default) --
  re-embedding the full corpus is required if switching between them, same
  caveat as noted in ADR-003.
- `gemini-embedding-2` (the newer multimodal model) was deliberately not
  used: it aggregates multiple inputs into a single embedding unless each
  input is wrapped in its own `Content` object, which is a more awkward fit
  for straightforward batch document embedding than `gemini-embedding-001`.
  Worth revisiting if multimodal embedding (e.g. embedding scanned PDF pages
  directly) becomes relevant later.
