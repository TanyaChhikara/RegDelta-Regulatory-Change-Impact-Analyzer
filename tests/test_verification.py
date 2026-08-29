"""Tests for src.analysis.verification."""

from src.analysis.gap_analysis import GapAnalysisResult
from src.analysis.verification import _best_match_ratio, verify_gap_analysis

REAL_POLICY_TEXT = """
4.1. Pursuant to RBI's temporary relaxation of the interest rate ceiling
applicable to certain long-tenor foreign-currency and NRE deposits, the Bank
may offer rates above the standard ceiling on fresh FCNR(B) deposits of 3-5
year tenors.

4.2. This temporary relaxation is effective from June 17, 2026, and remains
available for the period until September 30, 2026, in accordance with
RBI's direction dated November 28, 2025, as most recently updated.
"""

REAL_REGULATION_TEXT = """
Please refer to the Reserve Bank of India Directions, 2025. It has been
decided to amend the date September 30, 2026 to the date August 31, 2026
for the aforementioned temporary relaxation, with effect from June 17,
2026, for the period until August 31, 2026.
"""


def test_best_match_ratio_finds_real_quote_despite_embedded_newlines():
    claim = (
        "This temporary relaxation is effective from June 17, 2026, and "
        "remains available for the period until September 30, 2026"
    )
    ratio = _best_match_ratio(claim, REAL_POLICY_TEXT)
    assert ratio > 0.9


def test_best_match_ratio_scores_fabricated_claim_low():
    claim = "Section 9.4 requires quarterly board reporting on this matter"
    ratio = _best_match_ratio(claim, REAL_POLICY_TEXT)
    assert ratio < 0.5


def test_best_match_ratio_handles_empty_claim():
    assert _best_match_ratio("", REAL_POLICY_TEXT) == 0.0


def test_best_match_ratio_handles_empty_source():
    assert _best_match_ratio("some claim", "") == 0.0


def test_best_match_ratio_handles_claim_longer_than_source():
    long_claim = "a" * 500
    short_source = "short text"
    # Should not crash, should just return a low score.
    ratio = _best_match_ratio(long_claim, short_source)
    assert 0.0 <= ratio <= 1.0


def test_verify_gap_analysis_passes_real_grounded_claims():
    result = GapAnalysisResult(
        is_affected=True,
        confidence="High",
        reasoning="...",
        old_requirement="This temporary relaxation is effective from June 17, 2026, and "
        "remains available for the period until September 30, 2026",
        new_requirement="with effect from June 17, 2026, for the period until August 31, 2026",
        recommended_action="Update Section 4.2",
    )

    verification = verify_gap_analysis(result, REAL_REGULATION_TEXT, REAL_POLICY_TEXT)

    assert verification.old_requirement_grounded is True
    assert verification.new_requirement_grounded is True
    assert verification.fully_verified is True


def test_verify_gap_analysis_fails_fabricated_claims():
    result = GapAnalysisResult(
        is_affected=True,
        confidence="High",
        reasoning="...",
        old_requirement="Section 9.4 requires quarterly board reporting on this matter",
        new_requirement="The new mandatory capital buffer is 12.5% effective immediately",
        recommended_action="...",
    )

    verification = verify_gap_analysis(result, REAL_REGULATION_TEXT, REAL_POLICY_TEXT)

    assert verification.old_requirement_grounded is False
    assert verification.new_requirement_grounded is False
    assert verification.fully_verified is False


def test_verify_gap_analysis_not_affected_case_trivially_passes():
    result = GapAnalysisResult(
        is_affected=False,
        confidence="High",
        reasoning="No overlap found.",
        old_requirement=None,
        new_requirement=None,
        recommended_action=None,
    )

    verification = verify_gap_analysis(result, REAL_REGULATION_TEXT, REAL_POLICY_TEXT)

    assert verification.fully_verified is True


def test_verify_gap_analysis_detects_one_grounded_one_fabricated_claim():
    result = GapAnalysisResult(
        is_affected=True,
        confidence="High",
        reasoning="...",
        old_requirement="This temporary relaxation is effective from June 17, 2026, and "
        "remains available for the period until September 30, 2026",
        new_requirement="The new mandatory capital buffer is 12.5% effective immediately",
        recommended_action="...",
    )

    verification = verify_gap_analysis(result, REAL_REGULATION_TEXT, REAL_POLICY_TEXT)

    assert verification.old_requirement_grounded is True
    assert verification.new_requirement_grounded is False
    assert verification.fully_verified is False
