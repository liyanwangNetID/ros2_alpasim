import pytest

from step9.prompt import build_evidence_package
from step9.validator import validate_response


def source(with_longitudinal_support: bool = False):
    nodes = [
        {"node_id": "decision_lateral", "node_type": "lateral_decision", "value": {"action": "keep_direction", "quality_status": "usable"}, "source_ref": "x"},
        {"node_id": "decision_longitudinal", "node_type": "longitudinal_decision", "value": {"action": "decelerate", "quality_status": "usable"}, "source_ref": "x"},
        {"node_id": "navigation_intent", "node_type": "navigation_intent", "value": {"action": "straight", "quality_status": "usable"}, "source_ref": "x"},
    ]
    links = [
        {"link_id": "link_001", "source_node_id": "navigation_intent", "target_node_id": "decision_lateral", "relation": "aligns_with", "confidence": "supported", "rule_id": "navigation_lateral_alignment_v01", "reasons": ["compatible"]}
    ]
    if with_longitudinal_support:
        nodes.append({"node_id": "actor_lead_1", "node_type": "actor_state", "value": {"actor_class": "automobile", "relative_position": "front", "relative_distance": "near", "relative_speed_category": "slower_than_ego", "quality_status": "usable"}, "source_ref": "x"})
        links.append({"link_id": "link_002", "source_node_id": "actor_lead_1", "target_node_id": "decision_longitudinal", "relation": "supports", "confidence": "supported", "rule_id": "same_direction_lead_vehicle_supports_longitudinal_response_v01", "reasons": ["constraint"]})
    return {
        "anchor_id": "test_clip_001_1",
        "clip_id": "test_clip_001",
        "anchor_ns": 1,
        "quality": {"status": "usable", "reasons": []},
        "structured_coc": {"nodes": nodes, "links": links},
    }


def response(summary: str, keys=None):
    return {
        "reasoning_summary": summary,
        "used_evidence_keys": keys or ["navigation_alignment_01"],
        "limitations": [],
    }


@pytest.mark.parametrize(
    "summary",
    [
        "The longitudinal decision to decelerate is supported by the evidence.",
        "Decelerating is consistent with the available evidence.",
        "The vehicle slows down because of the evidence.",
        "Maintaining speed is consistent with the overall usable quality.",
    ],
)
def test_rejects_unsupported_longitudinal_claim(summary):
    package = build_evidence_package(source())
    with pytest.raises(ValueError):
        validate_response(response(summary), package)


def test_accepts_alignment_only_without_longitudinal_comment():
    package = build_evidence_package(source())
    value = response(
        "The straight navigation intent is compatible with keeping direction."
    )
    assert validate_response(value, package) == value


def test_accepts_longitudinal_claim_when_support_targets_longitudinal():
    package = build_evidence_package(source(with_longitudinal_support=True))
    value = response(
        "The straight navigation intent is compatible with keeping direction. A slower lead vehicle supports deceleration.",
        ["navigation_alignment_01", "same_direction_lead_vehicle_02"],
    )
    assert validate_response(value, package) == value
