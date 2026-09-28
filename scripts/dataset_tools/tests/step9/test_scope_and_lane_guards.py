import pytest
from step9.validator import validate_response


def package(targets=("lateral_decision",), relative_position=None):
    source = {"action": "straight", "quality_status": "usable"}
    if relative_position is not None:
        source = {"relative_position": relative_position, "lane_direction_relation": "same_direction", "quality_status": "usable"}
    return {"quality_status": "usable", "quality_reasons": [], "allowed_explanation_targets": list(targets), "evidence": [{"evidence_key": "evidence_01", "relation": "aligns_with", "confidence": "supported", "facts": {"source": source, "source_type": "actor_state" if relative_position else "navigation_intent", "target_type": targets[0]}}]}


def response(summary):
    return {"reasoning_summary": summary, "used_evidence_keys": ["evidence_01"], "limitations": []}


def test_rejects_longitudinal_uncertainty_outside_allowed_targets():
    with pytest.raises(ValueError, match="outside allowed explanation targets"):
        validate_response(response("There is insufficient evidence to determine the longitudinal action."), package())

@pytest.mark.parametrize("summary", ["A lead vehicle is in the front left lane.", "A vulnerable actor is in the front right lane."])
def test_rejects_lane_inference_from_relative_position(summary):
    with pytest.raises(ValueError, match="without lane evidence"):
        validate_response(response(summary), package(("longitudinal_decision",), "front_left"))

@pytest.mark.parametrize("summary", ["A lead vehicle is front left.", "A vulnerable actor is ahead and to the right."])
def test_allows_relative_position_without_lane_claim(summary):
    value = response(summary)
    assert validate_response(value, package(("longitudinal_decision",), "front_left")) == value

def test_allows_unknown_navigation_source_for_lateral_target():
    value = response("The navigation intent is unknown, so compatibility with the lateral decision cannot be confirmed.")
    assert validate_response(value, package()) == value


@pytest.mark.parametrize(
    "summary",
    [
        "Two lead vehicles are in the front right and front left lanes.",
        "Two lead vehicles are in the front left and front right lanes.",
        "Two lead vehicles are in the front right and left lanes.",
    ],
)
def test_rejects_plural_or_coordinated_lane_inference(summary):
    with pytest.raises(ValueError, match="without lane evidence"):
        validate_response(
            response(summary),
            package(("longitudinal_decision",), "front_right"),
        )
