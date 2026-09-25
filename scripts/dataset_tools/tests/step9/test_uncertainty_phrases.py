import pytest

from step9.validator import validate_response


def package():
    return {
        "quality_status": "unknown",
        "quality_reasons": ["meta_action_quality_unknown"],
        "evidence": [
            {
                "evidence_key": "navigation_evidence_insufficient_01",
                "relation": "insufficient_evidence",
                "confidence": "unknown",
                "facts": {"target_type": "lateral_decision"},
            }
        ],
    }


@pytest.mark.parametrize(
    "summary",
    [
        "The lateral action cannot be determined from the available evidence.",
        "There is not enough evidence to determine the lateral action.",
        "The explanation is constrained by a lack of evidence.",
        "No supported causal link is available for the lateral action.",
    ],
)
def test_accepts_clear_uncertainty_phrases(summary):
    value = {
        "reasoning_summary": summary,
        "used_evidence_keys": ["navigation_evidence_insufficient_01"],
        "limitations": ["meta_action_quality_unknown"],
    }
    assert validate_response(value, package()) == value
