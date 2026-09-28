import pytest

from step9.validator import validate_response


def partial_package():
    return {
        "quality_status": "partial",
        "quality_reasons": ["scene_facts_quality_unknown"],
        "evidence": [
            {
                "evidence_key": "navigation_alignment_01",
                "relation": "aligns_with",
                "confidence": "supported",
                "facts": {"target_type": "lateral_decision"},
            }
        ],
    }


@pytest.mark.parametrize(
    "summary",
    [
        "The evidence quality is partial, so the driving decisions should be interpreted with caution.",
        "The available evidence should be interpreted cautiously.",
        "The available evidence requires cautious interpretation.",
    ],
)
def test_partial_summary_accepts_caution_language(summary):
    value = {
        "reasoning_summary": summary,
        "used_evidence_keys": ["navigation_alignment_01"],
        "limitations": ["scene_facts_quality_unknown"],
    }
    assert validate_response(value, partial_package()) == value
