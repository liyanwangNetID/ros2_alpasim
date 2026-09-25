from __future__ import annotations

import json
from typing import Any, Mapping

SYSTEM_PROMPT = (
    "You generate a short, faithful driving-decision explanation from one structured "
    "evidence package. Return only the requested JSON object. Use only supplied facts and "
    "relations. Treat aligns_with only as compatibility between navigation and the lateral "
    "decision, never as causation. Treat supports as evidence for only the target decision "
    "shown in that evidence item, never as proof. Do not explain, justify, endorse, or call "
    "a longitudinal action supported unless a supports relation explicitly targets the "
    "longitudinal decision. If the only supported relation is navigation aligns_with the "
    "lateral decision, discuss only that compatibility and do not comment on accelerate, "
    "maintain_speed, decelerate, or stop. quality_status describes evidence quality only; "
    "it is never a reason for any driving action. For insufficient_evidence, partial "
    "quality, or unknown quality, explicitly state the uncertainty or evidence limitation "
    "in reasoning_summary. Do not output node IDs, link IDs, rule IDs, track IDs, hashes, "
    "file names, or provenance. Keep reasoning_summary to one or two concise sentences."
)


def evidence_key(index: int, relation: str, rule_id: str) -> str:
    prefixes = {
        "navigation_lateral_alignment_v01": "navigation_alignment",
        "navigation_lateral_evidence_insufficient_v01": "navigation_evidence_insufficient",
        "navigation_action_stage_uncertain_v01": "navigation_action_stage_uncertain",
        "same_direction_lead_vehicle_supports_longitudinal_response_v01": "same_direction_lead_vehicle",
        "front_vulnerable_actor_supports_longitudinal_response_v01": "front_vulnerable_actor",
    }
    return f"{prefixes.get(rule_id, relation)}_{index:02d}"


def build_evidence_package(row: Mapping[str, Any]) -> dict[str, Any]:
    nodes = {node["node_id"]: node for node in row["structured_coc"]["nodes"]}
    decisions: dict[str, Any] = {}
    node_specs = (
        ("lateral", "lateral_decision"),
        ("longitudinal", "longitudinal_decision"),
        ("navigation", "navigation_intent"),
    )
    for name, node_type in node_specs:
        node = next(item for item in nodes.values() if item["node_type"] == node_type)
        decisions[name] = dict(node["value"])

    evidence = []
    for index, link in enumerate(row["structured_coc"]["links"], start=1):
        source = nodes[link["source_node_id"]]
        target = nodes[link["target_node_id"]]
        facts = {
            "source_type": source["node_type"],
            "source": source["value"],
            "target_type": target["node_type"],
            "target": target["value"],
            "relation": link["relation"],
            "confidence": link["confidence"],
        }
        evidence.append(
            {
                "evidence_key": evidence_key(index, link["relation"], link["rule_id"]),
                "relation": link["relation"],
                "confidence": link["confidence"],
                "facts": facts,
            }
        )
    return {
        "record_id": row["anchor_id"],
        "decisions": decisions,
        "quality_status": row["quality"]["status"],
        "quality_reasons": row["quality"]["reasons"],
        "evidence": evidence,
    }


def user_prompt(package: Mapping[str, Any]) -> str:
    instruction = (
        "Generate the Step 9 response for this evidence package. Copy only evidence_key "
        "values actually used. Never explain a decision dimension that lacks evidence "
        "targeting that dimension."
    )
    payload = json.dumps(
        package,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return chr(10).join((instruction, payload))
