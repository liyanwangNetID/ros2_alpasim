from step9.prompt import build_evidence_package
from step9.trial import select_trial_rows, trial_tags


def make_row(anchor_id: str, quality: str, rule_id: str, relation: str = "aligns_with"):
    confidence = "supported" if relation != "insufficient_evidence" else "unknown"
    lateral_quality = "unknown" if quality == "unknown" else "usable"
    return {
        "anchor_id": anchor_id,
        "clip_id": "test_clip_001",
        "anchor_ns": 1,
        "quality": {
            "status": quality,
            "reasons": [] if quality == "usable" else ["limited_evidence"],
        },
        "structured_coc": {
            "nodes": [
                {"node_id": "decision_lateral", "node_type": "lateral_decision", "value": {"action": "keep_direction", "quality_status": lateral_quality}, "source_ref": "x"},
                {"node_id": "decision_longitudinal", "node_type": "longitudinal_decision", "value": {"action": "maintain_speed", "quality_status": "usable"}, "source_ref": "x"},
                {"node_id": "navigation_intent", "node_type": "navigation_intent", "value": {"action": "straight", "quality_status": "usable"}, "source_ref": "x"},
            ],
            "links": [
                {"link_id": "link_001", "source_node_id": "navigation_intent", "target_node_id": "decision_lateral", "relation": relation, "confidence": confidence, "rule_id": rule_id, "reasons": ["test"]}
            ],
        },
    }


def test_trial_tags_cover_quality_and_family():
    row = make_row("a", "partial", "navigation_lateral_evidence_insufficient_v01", "insufficient_evidence")
    tags = trial_tags(build_evidence_package(row))
    assert "quality:partial" in tags
    assert "family:navigation_evidence_insufficient" in tags


def test_trial_selection_is_deterministic_and_quality_stratified():
    rows = []
    for index in range(6):
        rows.append(make_row(f"u{index}", "usable", "navigation_lateral_alignment_v01"))
        rows.append(make_row(f"p{index}", "partial", "navigation_lateral_evidence_insufficient_v01", "insufficient_evidence"))
        rows.append(make_row(f"x{index}", "unknown", "navigation_lateral_evidence_insufficient_v01", "insufficient_evidence"))
    first = select_trial_rows(rows, limit=12)
    second = select_trial_rows(rows, limit=12)
    assert [row["anchor_id"] for row in first] == [row["anchor_id"] for row in second]
    qualities = [row["quality"]["status"] for row in first]
    assert qualities.count("usable") >= 4
    assert qualities.count("partial") >= 4
    assert qualities.count("unknown") >= 4
