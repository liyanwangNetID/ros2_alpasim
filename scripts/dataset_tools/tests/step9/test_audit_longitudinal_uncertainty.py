import pytest

from step9.validator import validate_response


def test_explicit_longitudinal_uncertainty_is_rejected_outside_allowed_targets():
    package = {
        "quality_status": "unknown",
        "quality_reasons": ["meta_action_quality_unknown"],
        "evidence": [
            {
                "evidence_key": "navigation_alignment_01",
                "relation": "aligns_with",
                "confidence": "supported",
                "facts": {"target_type": "lateral_decision"},
            }
        ],
    }
    response = {
        "reasoning_summary": "There is insufficient evidence to determine the appropriate longitudinal action.",
        "used_evidence_keys": ["navigation_alignment_01"],
        "limitations": ["meta_action_quality_unknown"],
    }
    with pytest.raises(ValueError, match="outside allowed explanation targets"):
        validate_response(response, package)
