# ADR-005: LLM Generation Provider for Gap Analysis

**Status:** Accepted
**Date:** 2026-08-30

## Context

M8 needs an LLM call to compare a regulation's text against a candidate
policy's text and produce a structured gap-analysis result (is the policy
affected, what changed, what should be reviewed). Anthropic does not offer
an embeddings API (per ADR-003/004), but does offer a generation API, so
generation and embeddings don't need to use the same provider.

Given the real payment friction already encountered getting OpenAI billing
working (ADR-004), and that Gemini's free tier already proved reliable for
embeddings, defaulting generation to a different provider than embeddings
would reintroduce the same card/billing risk for no clear benefit.

## Decision

**Default `LLM_PROVIDER` to Gemini, using `gemini-3.1-flash-lite`.**

## Reasoning

1. **Avoids re-introducing payment friction.** Same free-tier, no-card
   account already working for embeddings.
2. **Right-sized for the task.** Gap analysis here is closer to structured
   comparison/extraction (does clause X in the policy match clause Y in the
   regulation, what's the delta) than open-ended creative reasoning. Google
   explicitly positions `flash-lite`-tier models for "classification, simple
   extraction, tagging" -- a reasonable fit, and meaningfully cheaper than a
   full Pro-tier model for a task run repeatedly across many
   regulation-policy pairs.
3. **A pluggable provider (`LLM_PROVIDER`), same pattern as embeddings.**
   `anthropic` (Claude) and `fake` (deterministic, for dry runs / tests) are
   also supported, so switching is a config change.

## A caveat worth stating plainly

Gemini's model lineup changes fast -- during this same project, models
referenced in earlier research (`gemini-2.5-flash`) were already flagged
for shutdown by October 2026, barely months after release. `gemini-3.1-flash-lite`
is current and explicitly recommended by Google as of this writing, but
should be expected to need updating again within the lifetime of this
project. `GEMINI_GENERATION_MODEL` is a `.env` setting specifically so this
doesn't require a code change when that happens.

## Alternatives Considered

| Option | Verdict |
|---|---|
| Gemini `gemini-3.1-flash-lite` | **Chosen** -- free tier, no card, right-sized for structured comparison |
| Anthropic Claude | Supported as an alternative (`LLM_PROVIDER=anthropic`), not the default -- would require separate billing setup with its own potential friction |
| OpenAI GPT | Not implemented -- same billing friction already encountered with embeddings would apply here too |

## Consequences

- `.env.example` gains `LLM_PROVIDER`, `GEMINI_GENERATION_MODEL`, and
  (for the unused-by-default alternative) `ANTHROPIC_API_KEY` was already
  present from M0.
- Gap-analysis quality is only as good as `flash-lite`'s reasoning on this
  task. If evaluation later shows systematic reasoning errors (not just
  retrieval misses), upgrading to a stronger model for this specific call
  is a config change, not a redesign.
