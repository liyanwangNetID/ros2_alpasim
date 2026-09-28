from step9.prompt import build_evidence_package

def test_navigation_insufficient_hides_unreferenced_longitudinal():
    row = {"anchor_id":"test_clip_113_227164111209000","quality":{"status":"unknown","reasons":["meta_action_quality_unknown","no_supported_causal_link"]},"structured_coc":{"nodes":[{"node_id":"lat","node_type":"lateral_decision","value":{"action":"unknown","quality_status":"unknown"}},{"node_id":"lon","node_type":"longitudinal_decision","value":{"action":"maintain_speed","quality_status":"usable"}},{"node_id":"nav","node_type":"navigation_intent","value":{"action":"straight","quality_status":"usable"}}],"links":[{"source_node_id":"nav","target_node_id":"lat","relation":"insufficient_evidence","confidence":"unknown","rule_id":"navigation_lateral_evidence_insufficient_v01"}]}}
    package = build_evidence_package(row)
    assert set(package["decisions"]) == {"navigation", "lateral"}
    assert package["allowed_explanation_targets"] == ["lateral_decision"]
    assert "maintain_speed" not in str(package)
