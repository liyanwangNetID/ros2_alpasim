import pytest

from step9.validator import validate_response


def package():
    return {
        "quality_status": "usable",
        "quality_reasons": [],
        "evidence": [
            {
                "evidence_key": "navigation_alignment_01",
                "relation": "aligns_with",
                "confidence": "supported",
                "facts": {"target_type": "lateral_decision"},
            }
        ],
    }


def response(summary):
    return {
        "reasoning_summary": summary,
        "used_evidence_keys": ["navigation_alignment_01"],
        "limitations": [],
    }


@pytest.mark.parametrize(
    "summary",
    [
        "There is no evidence to suggest a need for a longitudinal action other than maintaining speed.",
        "There is no evidence to suggest the need for deceleration.",
    ],
)
def test_rejects_unsupported_longitudinal_default_inference(summary):
    with pytest.raises(ValueError, match="without longitudinal evidence"):
        validate_response(response(summary), package())


def test_rejects_explicit_longitudinal_uncertainty_outside_allowed_targets():
    value = response(
        "There is insufficient evidence to determine the appropriate longitudinal action."
    )
    with pytest.raises(ValueError, match="outside allowed explanation targets"):
        validate_response(value, package())
