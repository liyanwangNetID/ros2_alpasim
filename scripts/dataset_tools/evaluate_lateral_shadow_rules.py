#!/usr/bin/env python3
"""Compare proposed lane-change/turn arbitration against frozen v0.1 labels.

This tool is shadow-only. It never overwrites meta_actions_v0.1.jsonl.
It joins the frozen labels with full-scale lane-change geometry features and
writes proposed lateral actions plus an old-to-proposed transition matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

VERSION = "0.2.2"
from project_paths import ALPASIM_DATA_ROOT

ROOT = ALPASIM_DATA_ROOT
DEFAULT_LABEL_INPUT = ROOT / "annotations/v0.1-draft/meta_actions_v0.1.jsonl"
DEFAULT_GEOMETRY_INPUT = (
    ROOT / "annotations/v0.1-draft/intermediate/"
    "lane_change_geometry_features_v0.1.jsonl"
)
DEFAULT_LATERAL_FEATURE_INPUT = (
    ROOT / "annotations/v0.1-draft/intermediate/"
    "lateral_action_features_v0.3.jsonl"
)
DEFAULT_OUTPUT = ROOT / "reports/lateral_shadow_evaluation_v0.1.jsonl"
DEFAULT_SUMMARY = ROOT / "reports/lateral_shadow_evaluation_summary_v0.1.json"


def read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record["anchor_id"])
            if anchor_id in result:
                raise ValueError(f"duplicate anchor_id at {path}:{line_number}: {anchor_id}")
            result[anchor_id] = record
    return result


def atomic_write(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"output exists: {path}; use --force")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def normalize_interpretation(value: str) -> str | None:
    mapping = {
        "turn_left_candidate": "turn_left",
        "turn_right_candidate": "turn_right",
        "change_lane_left": "change_lane_left",
        "change_lane_right": "change_lane_right",
        "keep_direction": "keep_direction",
    }
    return mapping.get(value)


def observed_proposal(items: Sequence[Mapping[str, Any]]) -> tuple[str | None, list[str]]:
    proposals: list[tuple[str, str]] = []
    for item in items:
        normalized = normalize_interpretation(str(item.get("interpretation", "")))
        if normalized is None:
            continue
        proposals.append((normalized, str(item.get("interpretation_reason", ""))))
    if not proposals:
        return None, []
    actions = {action for action, _ in proposals}
    if len(actions) != 1:
        return "unknown", ["conflicting_observed_geometry_interpretations"]
    action = proposals[0][0]
    return action, sorted({reason for _, reason in proposals if reason})


def reviewed_in_progress_proposal(
    geometry: Mapping[str, Any],
    lateral_record: Mapping[str, Any] | None = None,
) -> tuple[str | None, list[str]]:
    actions: set[str] = set()

    # Backwards-compatible unit-test behavior when no lateral record is passed.
    # Production evaluation always supplies the strict third input.
    if lateral_record is not None:
        lateral = lateral_record.get("lateral", lateral_record)
        ego_change = lateral.get("ego_total_yaw_change_rad")
        map_change = lateral.get("map_corridor_heading_change_rad")

        if not isinstance(ego_change, (int, float)):
            return None, []
        if not isinstance(map_change, (int, float)):
            return None, []

        residual_deg = abs(float(ego_change) - float(map_change)) * 180.0 / 3.141592653589793
        if residual_deg < 2.0:
            return None, []

    for item in geometry.get("in_progress_candidates", []):
        if not bool(item.get("candidate")):
            continue

        direction = str(item.get("direction", ""))
        if direction not in {"left", "right"}:
            continue

        final_advantage = item.get("final_target_advantage_m")
        heading_progress = item.get("directional_heading_progress_deg")

        if not isinstance(final_advantage, (int, float)):
            continue
        if not isinstance(heading_progress, (int, float)):
            continue

        # Reviewed false positives were a curved-road case with very large
        # heading progress and an early candidate that remained too far from
        # the target corridor. The reviewed true lane changes stayed within
        # this envelope.
        if float(final_advantage) < -2.0:
            continue
        if abs(float(heading_progress)) > 10.0:
            continue

        actions.add(f"change_lane_{direction}")

    if not actions:
        return None, []
    if len(actions) > 1:
        return "unknown", ["conflicting_reviewed_in_progress_directions"]

    return next(iter(actions)), [
        "reviewed_in_progress_lane_change_geometry",
        "final_target_advantage_at_least_minus_2m",
        "directional_heading_progress_at_most_10deg",
        "absolute_ego_to_map_heading_residual_at_least_2deg",
    ]


def reviewed_lane_change_to_turn_allowed(
    old_action: str,
    proposed_action: str,
    geometry: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    if not old_action.startswith("change_lane_"):
        return True, []
    if not proposed_action.startswith("turn_"):
        return True, []

    expected_interpretation = proposed_action + "_candidate"

    for item in geometry.get("observed_adjacent_transitions", []):
        if str(item.get("interpretation", "")) != expected_interpretation:
            continue

        junction_level = str(item.get("junction_evidence_level", "C"))
        residual = item.get("source_heading_residual", {})
        ego_heading_change_deg = residual.get("ego_heading_change_deg")

        if junction_level not in {"A", "B"}:
            continue
        if not isinstance(ego_heading_change_deg, (int, float)):
            continue
        if abs(float(ego_heading_change_deg)) < 8.0:
            continue

        return True, [
            "reviewed_lane_change_to_turn_geometry",
            "junction_level_a_or_b",
            "post_transition_ego_heading_change_at_least_8deg",
        ]

    return False, [
        "lane_change_to_turn_review_gate_not_met",
        "requires_junction_level_a_or_b",
        "requires_post_transition_ego_heading_change_at_least_8deg",
    ]


def propose_lateral(
    frozen: Mapping[str, Any],
    geometry: Mapping[str, Any],
    lateral_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    old_action = str(frozen["lateral"]["action"])
    old_quality = str(frozen["lateral"].get("quality_status", "unknown"))

    if old_quality != "usable":
        return {
            "action": old_action,
            "decision_source": "preserve_frozen_unknown",
            "reasons": ["frozen_lateral_quality_not_usable"],
        }

    observed_action, observed_reasons = observed_proposal(
        geometry.get("observed_adjacent_transitions", [])
    )

    if observed_action is not None:
        # Regression-review policy: geometry may correct a frozen lane change
        # into an intersection turn, but may not downgrade it to keep_direction.
        if (
            old_action.startswith("change_lane_")
            and observed_action == "keep_direction"
        ):
            return {
                "action": old_action,
                "decision_source": "preserve_frozen_lane_change_against_keep_downgrade",
                "reasons": ["lane_change_to_keep_downgrade_disabled_after_review"],
            }

        turn_allowed, turn_gate_reasons = reviewed_lane_change_to_turn_allowed(
            old_action,
            observed_action,
            geometry,
        )
        if not turn_allowed:
            return {
                "action": old_action,
                "decision_source": "preserve_frozen_lane_change_against_weak_turn_revision",
                "reasons": turn_gate_reasons,
            }

        return {
            "action": observed_action,
            "decision_source": "observed_adjacent_geometry",
            "reasons": observed_reasons + turn_gate_reasons,
        }

    reviewed_action, reviewed_reasons = reviewed_in_progress_proposal(
        geometry,
        lateral_record,
    )

    if reviewed_action is not None:
        return {
            "action": reviewed_action,
            "decision_source": "reviewed_in_progress_lane_change_geometry",
            "reasons": reviewed_reasons,
        }

    return {
        "action": old_action,
        "decision_source": "preserve_frozen_label_no_new_evidence",
        "reasons": ["no_reviewed_geometry_rule_change"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-input", type=Path, default=DEFAULT_LABEL_INPUT)
    parser.add_argument("--geometry-input", type=Path, default=DEFAULT_GEOMETRY_INPUT)
    parser.add_argument(
        "--lateral-feature-input",
        type=Path,
        default=DEFAULT_LATERAL_FEATURE_INPUT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = read_jsonl_index(args.label_input)
    geometry = read_jsonl_index(args.geometry_input)
    lateral_features = read_jsonl_index(args.lateral_feature_input)

    label_ids = set(labels)
    geometry_ids = set(geometry)
    lateral_ids = set(lateral_features)

    if label_ids != geometry_ids or label_ids != lateral_ids:
        raise ValueError(
            "label, geometry, and lateral-feature anchor sets differ; "
            f"missing geometry={sorted(label_ids-geometry_ids)[:10]}, "
            f"extra geometry={sorted(geometry_ids-label_ids)[:10]}, "
            f"missing lateral={sorted(label_ids-lateral_ids)[:10]}, "
            f"extra lateral={sorted(lateral_ids-label_ids)[:10]}"
        )

    output: list[dict[str, Any]] = []
    transitions = Counter()
    source_counts = Counter()
    for anchor_id, frozen in labels.items():
        geo = geometry[anchor_id]
        lateral_record = lateral_features[anchor_id]

        for other_name, other in (
            ("geometry", geo),
            ("lateral", lateral_record),
        ):
            if (
                frozen["clip_id"] != other["clip_id"]
                or frozen["anchor_ns"] != other["anchor_ns"]
            ):
                raise ValueError(
                    f"identity mismatch for {anchor_id} in {other_name} input"
                )

        proposal = propose_lateral(
            frozen,
            geo,
            lateral_record,
        )
        old_action = str(frozen["lateral"]["action"])
        proposed_action = str(proposal["action"])
        changed = old_action != proposed_action
        transitions[(old_action, proposed_action)] += 1
        source_counts[proposal["decision_source"]] += 1
        output.append({
            "shadow_format_version": "0.1-draft",
            "shadow_evaluator_version": VERSION,
            "anchor_id": anchor_id,
            "clip_id": frozen["clip_id"],
            "anchor_ns": frozen["anchor_ns"],
            "old_lateral_action": old_action,
            "proposed_lateral_action": proposed_action,
            "changed": changed,
            "decision_source": proposal["decision_source"],
            "reasons": proposal["reasons"],
            "old_decision_stage": frozen["lateral"].get("decision_stage"),
            "observed_adjacent_transition_count": len(
                geo.get("observed_adjacent_transitions", [])
            ),
            "inferred_in_progress_action": geo.get("inferred_in_progress_action"),
        })

    matrix = [
        {"old_action": old, "proposed_action": proposed, "count": count}
        for (old, proposed), count in sorted(transitions.items())
    ]
    summary = {
        "shadow_evaluator_version": VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "labels": str(args.label_input),
            "geometry": str(args.geometry_input),
            "lateral_features": str(args.lateral_feature_input),
        },
        "reviewed_in_progress_policy": {
            "minimum_final_target_advantage_m": -2.0,
            "maximum_absolute_directional_heading_progress_deg": 10.0,
            "minimum_absolute_ego_to_map_heading_residual_deg": 2.0,
        },
        "reviewed_lane_change_to_turn_policy": {
            "allowed_junction_levels": ["A", "B"],
            "minimum_absolute_post_transition_ego_heading_change_deg": 8.0,
            "otherwise": "preserve_frozen_lane_change",
        },
        "anchor_count": len(output),
        "changed_count": sum(item["changed"] for item in output),
        "unchanged_count": sum(not item["changed"] for item in output),
        "transition_matrix": matrix,
        "decision_source_counts": dict(source_counts),
        "important_change_counts": {
            "lane_change_to_turn": sum(
                count for (old, proposed), count in transitions.items()
                if old.startswith("change_lane_") and proposed.startswith("turn_")
            ),
            "lane_change_to_keep": sum(
                count for (old, proposed), count in transitions.items()
                if old.startswith("change_lane_") and proposed == "keep_direction"
            ),
            "keep_to_lane_change": sum(
                count for (old, proposed), count in transitions.items()
                if old == "keep_direction" and proposed.startswith("change_lane_")
            ),
            "turn_changed": sum(
                count for (old, proposed), count in transitions.items()
                if old.startswith("turn_") and old != proposed
            ),
        },
    }
    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output
    )
    atomic_write(args.output, output_text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        args.force,
    )
    print("Output:", args.output)
    print("Summary:", args.summary_output)
    print("SHA-256:", hashlib.sha256(output_text.encode()).hexdigest())
    print("Anchors:", len(output))
    print("Changed:", summary["changed_count"])
    print("Important changes:", summary["important_change_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
