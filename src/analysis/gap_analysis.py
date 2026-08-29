"""
LLM-based gap analysis: given a regulation's text and a candidate policy's
text, determine whether the policy is actually affected, and if so,
specifically how -- the old requirement, the new requirement, and a
recommended action. This is the step that turns "these two documents are
semantically similar" (M8 part 2) into an actual evidence-first impact
assessment, per the project's evidence-first design principle: every claim
should be traceable to the specific regulatory and policy text that
supports it, not just an LLM's unstructured opinion.

Provider is pluggable (LLM_PROVIDER in .env): "gemini" (default, see
docs/adr/005-llm-generation-provider.md), "anthropic", or "fake" (dry runs,
no API call).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("gap_analysis")

GAP_ANALYSIS_PROMPT_TEMPLATE = """\
You are a regulatory compliance analyst. Compare the REGULATORY TEXT below \
against the INTERNAL POLICY TEXT below, and determine whether the policy is \
affected by this regulation, and if so, specifically how.

REGULATORY TEXT:
{regulation_text}

INTERNAL POLICY TEXT:
{policy_text}

Respond with ONLY a JSON object (no markdown code fences, no other text) \
with exactly these fields:
{{
  "is_affected": true or false,
  "confidence": "High", "Medium", or "Low",
  "reasoning": "1-3 sentence explanation of your determination",
  "old_requirement": "the specific outdated provision in the policy, quoted \
or closely paraphrased, or null if not affected",
  "new_requirement": "the specific provision in the regulation that \
supersedes it, quoted or closely paraphrased, or null if not affected",
  "recommended_action": "a specific, actionable recommendation (e.g. which \
section to update and to what), or null if not affected"
}}
"""


@dataclass
class GapAnalysisResult:
    is_affected: bool
    confidence: str
    reasoning: str
    old_requirement: str | None
    new_requirement: str | None
    recommended_action: str | None


_FAKE_RESULT = GapAnalysisResult(
    is_affected=True,
    confidence="Low",
    reasoning="Fake provider response for dry runs -- not a real analysis.",
    old_requirement="fake old requirement",
    new_requirement="fake new requirement",
    recommended_action="fake recommended action",
)


def _parse_llm_json(raw_text: str) -> GapAnalysisResult:
    """Parse the model's JSON response, tolerating markdown code fences some
    models add despite being asked not to.
    """
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    data = json.loads(cleaned)
    return GapAnalysisResult(
        is_affected=bool(data["is_affected"]),
        confidence=data.get("confidence", "Low"),
        reasoning=data.get("reasoning", ""),
        old_requirement=data.get("old_requirement"),
        new_requirement=data.get("new_requirement"),
        recommended_action=data.get("recommended_action"),
    )


def _call_gemini(prompt: str, model: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return response.text


def _call_anthropic(prompt: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def analyze_gap(regulation_text: str, policy_text: str) -> GapAnalysisResult:
    """Compare a regulation against a policy and return a structured gap analysis.

    On a malformed or unparseable LLM response, returns a Low-confidence
    "not affected" result with the parse failure recorded in `reasoning`,
    rather than raising -- a single bad response from one policy candidate
    shouldn't crash analysis of the others.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini")
    prompt = GAP_ANALYSIS_PROMPT_TEMPLATE.format(
        regulation_text=regulation_text, policy_text=policy_text
    )

    if provider == "fake":
        return _FAKE_RESULT

    if provider == "gemini":
        model = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.1-flash-lite")
        raw_response = _call_gemini(prompt, model)
    elif provider == "anthropic":
        model = os.getenv("ANTHROPIC_GENERATION_MODEL", "claude-haiku-4-5-20251001")
        raw_response = _call_anthropic(prompt, model)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use 'gemini', 'anthropic', or 'fake'."
        )

    try:
        return _parse_llm_json(raw_response)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Could not parse LLM response as expected JSON: %s", exc)
        return GapAnalysisResult(
            is_affected=False,
            confidence="Low",
            reasoning=f"LLM response could not be parsed: {exc}",
            old_requirement=None,
            new_requirement=None,
            recommended_action=None,
        )


def gap_analysis_to_dict(result: GapAnalysisResult) -> dict:
    return asdict(result)
