# ADR-003: Embedding Provider Choice

**Status:** Accepted
**Date:** 2026-08-30

## Context

Anthropic does not offer its own embeddings API, so any Claude-based RAG
system pairs with an external embedding provider regardless of which LLM
generates the final answer. A 2026 survey of current comparisons (StackAI,
TECHSY, Milvus, and others) shows the field has moved since older
recommendations defaulted unconditionally to OpenAI:

- **Voyage AI** (voyage-4 family, launched January 2026) leads on pure
  retrieval quality in several independent benchmarks, and is specifically
  called out as strong for code and technical/legal documentation --
  relevant to a regulatory-text corpus like this project's.
- **OpenAI's `text-embedding-3-small`** remains the standard cheap,
  well-documented starting point most RAG prototypes still begin with
  ($0.02/1M tokens), and is what most tutorials and the original technology
  shortlist assumed.
- Other strong options (Cohere, Gemini Embedding, BGE-M3 self-hosted) are
  more specialized toward multilingual or self-hosted use cases that don't
  apply here (the corpus is English-language RBI text; self-hosting isn't a
  goal for the MVP).

## Decision

**Default to OpenAI `text-embedding-3-small` for the MVP, behind a pluggable
provider interface (`EMBEDDING_PROVIDER` in `.env`), so switching to Voyage
AI later is a configuration change, not a rewrite.**

## Reasoning

1. **Cost-consciousness for a learning project.** `text-embedding-3-small`
   is inexpensive enough to iterate freely without worrying about API
   spend, which matters more at this MVP stage than squeezing out the last
   few points of retrieval quality.
2. **Voyage AI is a legitimate V2 experiment, not a foregone default.**
   Whether Voyage's benchmarked advantage on technical/legal text actually
   shows up on *this* corpus (short RBI notifications, not long technical
   manuals) is an empirical question worth testing later, in the spirit of
   the project's "what hypothesis are we testing" approach to adding
   sophistication -- not something to assume upfront.
3. **The abstraction cost is low.** `embed_texts()` is the only function
   the rest of the pipeline depends on; both providers just need to return
   `list[list[float]]` for a list of input strings.

## Alternatives Considered

| Provider | Verdict |
|---|---|
| OpenAI `text-embedding-3-small` | **Chosen for MVP** -- cheap, standard, sufficient to validate the pipeline |
| Voyage AI (voyage-4) | Deferred -- worth an explicit quality experiment in V2, given claimed strength on technical/legal text |
| Cohere embed-v4 | Deferred -- multilingual strength doesn't apply (English-only corpus) |
| Self-hosted (BGE-M3) | Deferred -- adds GPU/infrastructure complexity with no current need |

## Consequences

- `EMBEDDING_PROVIDER=openai` (default) or `EMBEDDING_PROVIDER=fake` (for
  dry runs / testing without API cost) are supported now.
  `EMBEDDING_PROVIDER=voyage` is a planned addition, not yet implemented.
- A V2 experiment worth running once the evaluation framework exists:
  re-embed the corpus with Voyage AI and compare Recall@K against the
  OpenAI baseline on the same retrieval test set.
- Switching providers requires re-embedding the entire corpus -- embeddings
  from different models/providers are not compatible with each other and
  can't be mixed in one vector store.
