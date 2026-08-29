# ADR-006: Deterministic Evidence Verification Instead of a Verification Agent

**Status:** Accepted
**Date:** 2026-08-30

## Context

The original project specification sketched a Retrieval / Analysis /
Verification agent split, with the Verification agent "responsible for
checking whether conclusions are actually supported by evidence." M8 built
Retrieval (deterministic, `find_candidate_policies`) and Analysis (a single
LLM call, `analyze_gap`). Real runs against the live corpus (see the
`RBI/2026-27/243` analysis) show `analyze_gap` performing well -- but only
against genuinely relevant candidates (similarity 0.70-0.77). Its behavior
against weak or irrelevant candidates, where it might fabricate a
plausible-sounding but nonexistent quote, hasn't been tested.

The project's own stated philosophy (Section 10 of the original spec, and
this session's operating instructions) requires answering "why does this
need to be a separate agent?" before adding one, and using a deterministic
function instead whenever one suffices.

## Decision

**Implement evidence verification as a deterministic fuzzy-text-matching
function, not a second LLM call or an autonomous agent.**

`verify_gap_analysis()` checks whether `analyze_gap()`'s claimed
`old_requirement` text actually appears (closely, allowing for minor
paraphrasing) in the policy text it's attributed to, and whether
`new_requirement` appears in the regulation text. A claim that doesn't match
anything in its claimed source is flagged as unverified.

## Reasoning

1. **This is a string-matching problem, not a reasoning problem.** "Does
   this quote appear in this text" doesn't need an LLM's judgment -- a
   fuzzy similarity match (via Python's stdlib `difflib`) against the
   source text's sentences answers it directly, deterministically, and for
   free.
2. **An LLM verifier could itself hallucinate.** Asking a second LLM call
   "does this evidence check out?" introduces a new opportunity for the
   *verifier* to be wrong, without eliminating the original risk. A
   deterministic check has no such failure mode -- it either finds a
   sufficiently close match or it doesn't.
3. **No added cost or latency for a real behavioral guarantee.** Every
   `analyze_gap()` call already happens; verification runs in microseconds
   afterward with zero additional API calls.
4. **Matches the project's explicit anti-over-engineering stance.** The
   "Verification Agent" *role* from the original spec is fulfilled; an
   actual autonomous agent is not, because nothing here requires autonomy,
   tool selection, or multi-step planning -- it's a single deterministic
   check applied twice per gap-analysis result.

## What would justify revisiting this

If real evaluation (analogous to M6.5, but for gap analysis) finds cases
this deterministic check can't handle -- e.g., a genuinely correct claim
that's paraphrased heavily enough to score low on fuzzy matching (a false
"unverified" flag), or a hallucination clever enough to closely match
unrelated real text (a false "verified" pass) -- that would be real,
evidence-based justification for an LLM-based verifier as a targeted
upgrade, not a default assumption carried over from the original spec.

## Alternatives Considered

| Option | Verdict |
|---|---|
| Deterministic fuzzy-match verification | **Chosen** -- free, fast, no new hallucination surface |
| Second LLM call asking "is this evidence real?" | Rejected -- solves a string-matching problem with a more expensive, less reliable tool |
| Full autonomous Verification agent (tool use, planning loop) | Rejected -- no task here requires autonomy; would be complexity without a corresponding need |

## Consequences

- `src/analysis/verification.py` adds no new API dependency or cost.
- The CLI (`analyze_regulation.py`) surfaces an explicit warning when a
  claim doesn't verify, rather than silently trusting every LLM output.
- This doesn't preclude a real multi-agent architecture later for a task
  that genuinely needs one (e.g., iterative multi-hop retrieval across
  Master Directions once those are ingested) -- it specifically means this
  particular role doesn't need it.
