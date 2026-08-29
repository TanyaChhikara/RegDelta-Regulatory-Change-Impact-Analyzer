"""Tests for src.analysis.gap_analysis."""

from unittest.mock import MagicMock, patch

from src.analysis.gap_analysis import (
    GapAnalysisResult,
    _parse_llm_json,
    analyze_gap,
    gap_analysis_to_dict,
)

VALID_JSON_RESPONSE = """{
  "is_affected": true,
  "confidence": "High",
  "reasoning": "The policy cites the pre-amendment deadline.",
  "old_requirement": "period until September 30, 2026",
  "new_requirement": "period until August 31, 2026",
  "recommended_action": "Update Section 4.2 date."
}"""


def test_parse_llm_json_handles_clean_json():
    result = _parse_llm_json(VALID_JSON_RESPONSE)
    assert result.is_affected is True
    assert result.confidence == "High"
    assert result.old_requirement == "period until September 30, 2026"


def test_parse_llm_json_strips_markdown_code_fences():
    fenced = f"```json\n{VALID_JSON_RESPONSE}\n```"
    result = _parse_llm_json(fenced)
    assert result.is_affected is True
    assert result.new_requirement == "period until August 31, 2026"


def test_parse_llm_json_handles_not_affected_case():
    raw = """{
      "is_affected": false,
      "confidence": "Medium",
      "reasoning": "No overlap between the two documents.",
      "old_requirement": null,
      "new_requirement": null,
      "recommended_action": null
    }"""
    result = _parse_llm_json(raw)
    assert result.is_affected is False
    assert result.old_requirement is None
    assert result.recommended_action is None


def test_analyze_gap_fake_provider_returns_fixed_result(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    result = analyze_gap("regulation text", "policy text")
    assert isinstance(result, GapAnalysisResult)
    assert "Fake provider" in result.reasoning


def test_analyze_gap_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not_a_real_provider")
    try:
        analyze_gap("a", "b")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "Unknown LLM_PROVIDER" in str(e)


def test_analyze_gap_gemini_provider_calls_api_and_parses_response(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "gemini-3.1-flash-lite")

    mock_response = MagicMock(text=VALID_JSON_RESPONSE)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = analyze_gap("regulation text", "policy text")

    assert result.is_affected is True
    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["model"] == "gemini-3.1-flash-lite"
    assert "regulation text" in kwargs["contents"]
    assert "policy text" in kwargs["contents"]


def test_analyze_gap_anthropic_provider_calls_api_and_parses_response(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_GENERATION_MODEL", "claude-haiku-4-5-20251001")

    mock_content_block = MagicMock(text=VALID_JSON_RESPONSE)
    mock_response = MagicMock(content=[mock_content_block])
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = analyze_gap("regulation text", "policy text")

    assert result.is_affected is True
    _, kwargs = mock_client.messages.create.call_args
    assert kwargs["model"] == "claude-haiku-4-5-20251001"


def test_analyze_gap_handles_malformed_response_gracefully(monkeypatch):
    """A single bad LLM response should degrade to a Low-confidence
    not-affected result, not crash the whole analysis run.
    """
    monkeypatch.setenv("LLM_PROVIDER", "gemini")

    mock_response = MagicMock(text="This is not valid JSON at all.")
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        result = analyze_gap("regulation text", "policy text")

    assert result.is_affected is False
    assert result.confidence == "Low"
    assert "could not be parsed" in result.reasoning


def test_gap_analysis_to_dict_returns_plain_dict():
    result = GapAnalysisResult(
        is_affected=True, confidence="High", reasoning="r",
        old_requirement="old", new_requirement="new", recommended_action="act",
    )
    d = gap_analysis_to_dict(result)
    assert d == {
        "is_affected": True, "confidence": "High", "reasoning": "r",
        "old_requirement": "old", "new_requirement": "new", "recommended_action": "act",
    }
