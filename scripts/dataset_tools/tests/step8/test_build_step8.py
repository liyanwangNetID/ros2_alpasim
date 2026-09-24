from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from step8.build_step8 import (
    canonical_json,
    output_paths,
    structured_causality_schema,
    summarize,
    validate_record_invariants,
)
from step8.contract import Step8Paths


def record(*, anchor="test_clip_001_10", relation="aligns_with", actor=False):
    nodes = [
        {"node_id": "decision_lateral", "node_type": "lateral_decision", "value": {"action": "keep_direction"}, "source_ref": "meta_actions_v0.2.jsonl"},
        {"node_id": "decision_longitudinal", "node_type": "longitudinal_decision", "value": {"action": "decelerate"}, "source_ref": "meta_actions_v0.2.jsonl"},
        {"node_id": "navigation_intent", "node_type": "navigation_intent", "value": {"action": "straight"}, "source_ref": "navigation.jsonl"},
    ]
    source = "navigation_intent"
    target = "decision_lateral"
    rule = "navigation_lateral_alignment_v01"
    if actor:
        nodes.append({"node_id": "actor_lead_1", "node_type": "actor_state", "value": {"lane_direction_relation": "same_direction"}, "source_ref": "scene_facts.jsonl"})
        source = "actor_lead_1"
        target = "decision_longitudinal"
        relation = "supports"
        rule = "same_direction_lead_vehicle_supports_longitudinal_response_v01"
    return {
        "structured_causality_format_version": "0.1-draft",
        "generator_version": "0.1.0",
        "rule_version": "structured_causality_rules_v0.2",
        "anchor_id": anchor,
        "clip_id": "test_clip_001",
        "anchor_ns": 10,
        "structured_coc": {
            "nodes": sorted(nodes, key=lambda item: item["node_id"]),
            "links": [{
                "link_id": "link_001",
                "source_node_id": source,
                "target_node_id": target,
                "relation": relation,
                "confidence": "supported",
                "rule_id": rule,
                "reasons": ["test_evidence"],
            }],
        },
        "evidence_refs": {
            "navigation_anchor_id": anchor,
            "scene_fact_anchor_id": anchor,
            "meta_action_anchor_id": anchor,
        },
        "quality": {
            "status": "usable",
            "source_quality": {"navigation": "usable", "scene_facts": "usable", "meta_action": "usable"},
            "reasons": [],
        },
        "source_versions": {
            "navigation_format_version": "0.1-draft",
            "scene_fact_format_version": "0.1-draft",
            "meta_action_format_version": "0.2-draft",
        },
    }


def paths(root: Path) -> Step8Paths:
    annotations = root / "annotations" / "v0.1-draft"
    return Step8Paths(
        data_root=root,
        keyframes=annotations / "keyframes.jsonl",
        meta_actions=annotations / "meta_actions_v0.2.jsonl",
        navigation=annotations / "navigation.jsonl",
        scene_facts=annotations / "scene_facts.jsonl",
        keyframe_contract=root / "manifests" / "keyframe_contract_v0.1.json",
    )


def test_schema_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(structured_causality_schema())


def test_schema_accepts_representative_record():
    errors = list(Draft202012Validator(structured_causality_schema()).iter_errors(record()))
    assert errors == []


def test_invariants_accept_closed_graph():
    validate_record_invariants(record(actor=True))


def test_invariants_reject_dangling_link():
    value = record()
    value["structured_coc"]["links"][0]["source_node_id"] = "missing"
    with pytest.raises(ValueError, match="dangling"):
        validate_record_invariants(value)


def test_invariants_reject_unlinked_actor_node():
    value = record()
    value["structured_coc"]["nodes"].append({"node_id": "actor_x", "node_type": "actor_state", "value": {}, "source_ref": "scene_facts.jsonl"})
    with pytest.raises(ValueError, match="unlinked Actor"):
        validate_record_invariants(value)


def test_summary_counts_records_relations_and_actor_nodes():
    value = summarize([record(), record(anchor="test_clip_001_20", actor=True)])
    assert value["record_count"] == 2
    assert value["relation_counts"] == {"aligns_with": 1, "supports": 1}
    assert value["actor_node_count_distribution"] == {"0": 1, "1": 1}


def test_canonical_json_is_byte_stable():
    first = canonical_json({"b": 2, "a": 1})
    second = canonical_json({"a": 1, "b": 2})
    assert first == second == '{"a":1,"b":2}'


def test_output_paths_follow_data_root(tmp_path):
    value = output_paths(paths(tmp_path))
    assert value["output"] == tmp_path / "annotations" / "v0.1-draft" / "structured_causality.jsonl"
    assert value["schema"] == tmp_path / "schemas" / "structured_causality_schema_v0.1-draft.json"
    assert value["summary"] == tmp_path / "reports" / "step8_structured_causality_summary_v01.json"
    assert value["contract"] == tmp_path / "manifests" / "structured_causality_contract_v0.1.json"
