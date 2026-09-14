#!/usr/bin/env python3
"""Generate finalized Step 6 v0.1 Navigation records."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from project_paths import (
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    INTERMEDIATE_ROOT,
    MANIFEST_ROOT,
    REPORT_ROOT,
)

RULE_VERSION = "navigation_rules_v0.1.4"


GENERATOR_VERSION = "0.1.4"


OUTPUT_FORMAT_VERSION = "0.1-draft"


UPCOMING_TIME_HORIZON_SEC = 10.0


MINIMUM_UPCOMING_DISTANCE_M = 15.0


MAXIMUM_UPCOMING_DISTANCE_M = 80.0


DIRECTION_GEOMETRY_MINIMUM_DEG = 5.0


ROAD_LEVEL_INTERSECTION_DIRECTION_THRESHOLD_DEG = 20.0


VALID_ACTIONS = {"straight", "left", "right", "unknown"}


def _unknown(*reasons: str) -> dict[str, Any]:
    return {
        "action": "unknown",
        "text": None,
        "quality_status": "unknown",
        "decision_source": "insufficient_route_or_map_evidence",
        "reasons": list(dict.fromkeys(reasons)),
    }


def dynamic_upcoming_distance_m(
    *, current_speed_mps: float, route_lookahead_distance_m: float,
) -> float:
    speed = max(0.0, float(current_speed_mps))
    route_limit = max(0.0, float(route_lookahead_distance_m))
    configured = max(
        MINIMUM_UPCOMING_DISTANCE_M,
        speed * UPCOMING_TIME_HORIZON_SEC,
    )
    return min(route_limit, MAXIMUM_UPCOMING_DISTANCE_M, configured)


def _is_upcoming_intersection(
    context: Mapping[str, Any], *, upcoming_distance_m: float,
) -> bool:
    evidence = context.get("first_intersection_evidence")
    if not isinstance(evidence, Mapping):
        return False
    distance = evidence.get("route_distance_m")
    if not isinstance(distance, (int, float)):
        return False
    reasons = {str(value) for value in evidence.get("evidence", [])}
    return bool(reasons & {"wait_line", "branching", "merging"}) and float(distance) <= upcoming_distance_m


def classify_navigation(
    route_features: Mapping[str, Any],
    branch_features: Mapping[str, Any],
    road_level_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if str(route_features.get("quality_status")) != "usable":
        return _unknown("navigation_route_geometry_not_usable")
    if str(branch_features.get("quality_status")) != "usable":
        return _unknown("navigation_branch_context_not_usable")

    context = branch_features.get("branch_context", {})
    route_geometry = route_features.get("route", {})
    speed = branch_features.get("anchor_speed_mps")
    lookahead = route_features.get("route_lookahead_distance_m")
    if not isinstance(context, Mapping) or not isinstance(route_geometry, Mapping):
        return _unknown("navigation_context_missing")
    if not isinstance(speed, (int, float)) or not isinstance(lookahead, (int, float)):
        return _unknown("anchor_speed_or_route_lookahead_missing")

    upcoming_distance = dynamic_upcoming_distance_m(
        current_speed_mps=float(speed),
        route_lookahead_distance_m=float(lookahead),
    )
    upcoming_intersection = _is_upcoming_intersection(
        context, upcoming_distance_m=upcoming_distance,
    )
    branch = context.get("first_observed_branch")

    if branch is None:
        return {
            "action": "straight",
            "text": "Continue straight through the upcoming intersection." if upcoming_intersection else "Continue along the road.",
            "quality_status": "usable",
            "decision_source": "route_no_observed_branch_upcoming_intersection" if upcoming_intersection else "route_no_observed_branch",
            "reasons": ["route_follows_observed_lane_sequence_without_branch_choice"],
        }
    if not isinstance(branch, Mapping):
        return _unknown("first_observed_branch_invalid")

    relation = str(branch.get("route_relation_to_natural", ""))
    reliability = str(branch.get("reliability_status", ""))
    branch_distance = branch.get("route_distance_m")
    if not isinstance(branch_distance, (int, float)):
        return _unknown("first_observed_branch_distance_missing")

    if float(branch_distance) > upcoming_distance:
        return {
            "action": "straight",
            "text": "Continue along the road.",
            "quality_status": "usable",
            "decision_source": "first_observed_branch_beyond_dynamic_preview_horizon",
            "reasons": [
                "route_branch_not_yet_within_navigation_preview_horizon"
            ],
        }

    if relation == "actual_successor_not_candidate":
        return _unknown("route_successor_not_in_branch_candidates")
    if relation == "natural_continuation":
        natural_successor = branch.get(
            "natural_successor_lane_id"
        )
        route_successor = branch.get(
            "route_successor_lane_id"
        )

        if (
            natural_successor is not None
            and route_successor is not None
            and str(natural_successor) != str(route_successor)
        ):
            return _unknown(
                "inconsistent_natural_continuation_successor_identity"
            )

        if not upcoming_intersection:
            return {
                "action": "straight",
                "text": "Continue along the road.",
                "quality_status": "usable",
                "decision_source": "route_natural_continuation",
                "reasons": ["route_selects_natural_successor", "no_upcoming_intersection"],
            }
        road_geometry = (
            road_level_features.get("road_level_route_geometry", {})
            if isinstance(road_level_features, Mapping)
            else {}
        )
        road_change = (
            road_geometry.get("route_road_level_heading_change_deg")
            if isinstance(road_geometry, Mapping)
            and str(road_geometry.get("status")) == "available"
            else None
        )
        if isinstance(road_change, (int, float)):
            if float(road_change) > ROAD_LEVEL_INTERSECTION_DIRECTION_THRESHOLD_DEG:
                action = "left"
            elif float(road_change) < -ROAD_LEVEL_INTERSECTION_DIRECTION_THRESHOLD_DEG:
                action = "right"
            else:
                if reliability != "reliable":
                    return _unknown(
                        "first_observed_branch_not_reliable",
                        "road_level_direction_below_intersection_threshold",
                    )
                action = "straight"
            return {
                "action": action,
                "text": (
                    "Continue straight through the upcoming intersection."
                    if action == "straight"
                    else f"Turn {action} at the upcoming intersection."
                ),
                "quality_status": "usable",
                "decision_source": f"road_level_natural_continuation_intersection_{action}",
                "reasons": [
                    "route_selects_natural_successor",
                    "upcoming_intersection_detected",
                    "road_level_direction_available",
                ],
            }
        if reliability != "reliable":
            return _unknown(
                "first_observed_branch_not_reliable",
                "road_level_direction_geometry_not_available",
            )
        return {
            "action": "straight",
            "text": "Continue straight through the upcoming intersection.",
            "quality_status": "usable",
            "decision_source": "route_natural_continuation_upcoming_intersection_legacy_fallback",
            "reasons": [
                "route_selects_natural_successor",
                "road_level_direction_geometry_not_available",
            ],
        }
    if reliability != "reliable":
        return _unknown("first_observed_branch_not_reliable")

    signed_change = route_geometry.get("route_signed_heading_change_rad")
    if not isinstance(signed_change, (int, float)):
        return _unknown("route_direction_geometry_missing")
    signed_deg = math.degrees(float(signed_change))

    if relation == "left_of_natural":
        if signed_deg < DIRECTION_GEOMETRY_MINIMUM_DEG:
            return _unknown(
                "left_branch_route_geometry_not_consistently_left"
                if signed_deg < -DIRECTION_GEOMETRY_MINIMUM_DEG
                else "left_branch_route_geometry_too_weak"
            )
        action = "left"
    elif relation == "right_of_natural":
        if signed_deg > -DIRECTION_GEOMETRY_MINIMUM_DEG:
            return _unknown(
                "right_branch_route_geometry_not_consistently_right"
                if signed_deg > DIRECTION_GEOMETRY_MINIMUM_DEG
                else "right_branch_route_geometry_too_weak"
            )
        action = "right"
    else:
        return _unknown("first_observed_branch_relation_unresolved")

    if upcoming_intersection:
        return {
            "action": action,
            "text": f"Turn {action} at the upcoming intersection.",
            "quality_status": "usable",
            "decision_source": f"route_intersection_{action}_branch",
            "reasons": ["upcoming_intersection_detected", f"route_selects_{action}_branch", "route_geometry_direction_consistent"],
        }
    return {
        "action": action,
        "text": f"Follow the {action} branch ahead.",
        "quality_status": "usable",
        "decision_source": f"route_{action}_branch",
        "reasons": [f"route_selects_{action}_branch", "route_geometry_direction_consistent"],
    }


ROOT = ALPASIM_DATA_ROOT
ANN = ANNOTATION_ROOT
DEFAULT_KEYFRAMES = ANNOTATION_ROOT / "keyframes.jsonl"
DEFAULT_ROUTE_FEATURES = (
    INTERMEDIATE_ROOT
    / "navigation_route_features_v0.1.jsonl"
)
DEFAULT_BRANCH_FEATURES = (
    INTERMEDIATE_ROOT
    / "navigation_branch_context_v0.1.jsonl"
)
DEFAULT_ROAD_LEVEL_FEATURES = (
    INTERMEDIATE_ROOT
    / "road_level_navigation_features_v0.1.jsonl"
)
DEFAULT_OUTPUT = (
    ANNOTATION_ROOT / "navigation.jsonl"
)
DEFAULT_SUMMARY = (
    REPORT_ROOT / "navigation_generation_summary_v0.1.json"
)
DEFAULT_KEYFRAME_CONTRACT = (
    MANIFEST_ROOT / "keyframe_contract_v0.1.json"
)
EXPECTED_KEYFRAME_CONTRACT_VERSION = "0.1"
EXPECTED_KEYFRAME_PRODUCER_STEP = 5


def read_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record["anchor_id"])
            if anchor_id in result:
                raise ValueError(f"duplicate Anchor at {path}:{line_number}: {anchor_id}")
            result[anchor_id] = record
    return result



def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"required JSON file not found: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in {path}: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise ValueError(
            f"JSON root must be an object: {path}"
        )

    return value


def validate_keyframe_contract(
    contract_path: Path,
    keyframe_path: Path,
    keyframe_ids: set[str],
) -> dict[str, Any]:
    contract = read_json_object(contract_path)

    if (
        str(contract.get("contract_version"))
        != EXPECTED_KEYFRAME_CONTRACT_VERSION
    ):
        raise ValueError(
            "unsupported Keyframe contract version: "
            f"{contract.get('contract_version')}"
        )

    if (
        contract.get("producer_step")
        != EXPECTED_KEYFRAME_PRODUCER_STEP
    ):
        raise ValueError(
            "Keyframe contract producer_step must be "
            f"{EXPECTED_KEYFRAME_PRODUCER_STEP}"
        )

    try:
        keyframe_data = keyframe_path.read_bytes()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Keyframe file not found: {keyframe_path}"
        ) from exc

    actual_digest = hashlib.sha256(
        keyframe_data
    ).hexdigest()
    expected_digest = str(
        contract.get("keyframe_sha256", "")
    )

    if actual_digest != expected_digest:
        raise ValueError(
            "Keyframe SHA-256 mismatch: "
            f"{actual_digest} != {expected_digest}"
        )

    expected_count = int(
        contract["keyframe_count"]
    )

    if len(keyframe_ids) != expected_count:
        raise ValueError(
            "Keyframe count mismatch: "
            f"{len(keyframe_ids)} != {expected_count}"
        )

    return contract


def require_identity(anchor_id: str, reference: Mapping[str, Any], *others: Mapping[str, Any]) -> None:
    for record in others:
        for key in ("anchor_id", "clip_id", "anchor_ns"):
            if reference.get(key) != record.get(key):
                raise ValueError(f"{anchor_id}: identity mismatch for {key}")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframe-input", type=Path, default=DEFAULT_KEYFRAMES)
    parser.add_argument("--route-feature-input", type=Path, default=DEFAULT_ROUTE_FEATURES)
    parser.add_argument("--branch-feature-input", type=Path, default=DEFAULT_BRANCH_FEATURES)
    parser.add_argument("--road-level-feature-input", type=Path, default=DEFAULT_ROAD_LEVEL_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--keyframe-contract",
        type=Path,
        default=DEFAULT_KEYFRAME_CONTRACT,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    keyframes = read_index(args.keyframe_input)
    route_features = read_index(args.route_feature_input)
    branch_features = read_index(args.branch_feature_input)
    road_level_features = read_index(args.road_level_feature_input)
    expected = set(keyframes)

    keyframe_contract = validate_keyframe_contract(
        args.keyframe_contract,
        args.keyframe_input,
        expected,
    )

    if (
        set(route_features) != expected
        or set(branch_features) != expected
        or set(road_level_features) != expected
    ):
        raise ValueError(
            "Navigation feature Anchor sets do not match "
            "Keyframes; "
            f"route_missing="
            f"{len(expected - set(route_features))}, "
            f"route_extra="
            f"{len(set(route_features) - expected)}, "
            f"branch_missing="
            f"{len(expected - set(branch_features))}, "
            f"branch_extra="
            f"{len(set(branch_features) - expected)}, "
            f"road_level_missing="
            f"{len(expected - set(road_level_features))}, "
            f"road_level_extra="
            f"{len(set(road_level_features) - expected)}"
        )

    output = []
    action_counts: Counter[str] = Counter()
    text_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    unknown_reason_counts: Counter[str] = Counter()

    for anchor_id in sorted(
        expected,
        key=lambda value: (
            str(keyframes[value]["clip_id"]),
            int(keyframes[value]["anchor_ns"]),
            value,
        ),
    ):
        keyframe = keyframes[anchor_id]
        route = route_features[anchor_id]
        branch = branch_features[anchor_id]
        road_level = road_level_features[anchor_id]
        require_identity(anchor_id, keyframe, route, branch, road_level)
        navigation = classify_navigation(route, branch, road_level)
        if navigation["action"] not in VALID_ACTIONS:
            raise ValueError(f"{anchor_id}: invalid Navigation action")
        if navigation["action"] == "unknown" and navigation["text"] is not None:
            raise ValueError(f"{anchor_id}: unknown Navigation must use null text")

        record = {
            "navigation_format_version": OUTPUT_FORMAT_VERSION,
            "generator_version": GENERATOR_VERSION,
            "rule_version": RULE_VERSION,
            "anchor_id": anchor_id,
            "clip_id": str(keyframe["clip_id"]),
            "anchor_ns": int(keyframe["anchor_ns"]),
            "navigation": navigation,
            "source_versions": {
                "route_feature_format_version": route.get("feature_format_version"),
                "branch_context_format_version": branch.get("feature_format_version"),
                "road_level_feature_format_version": road_level.get("feature_format_version"),
            },
            "review_status": "reviewed_v0.1",
        }
        output.append(record)
        action_counts[navigation["action"]] += 1
        quality_counts[navigation["quality_status"]] += 1
        source_counts[navigation["decision_source"]] += 1
        text_counts[str(navigation["text"])] += 1
        if navigation["action"] == "unknown":
            unknown_reason_counts.update(navigation["reasons"])

    output_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in output
    )
    sha256 = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
    summary = {
        "navigation_format_version": OUTPUT_FORMAT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "rule_version": RULE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_status": "reviewed_v0.1",
        "input_keyframe_count": len(expected),
        "keyframe_contract": str(
            args.keyframe_contract
        ),
        "keyframe_sha256": str(
            keyframe_contract["keyframe_sha256"]
        ),
        "keyframe_count": int(
            keyframe_contract["keyframe_count"]
        ),
        "keyframe_coverage_valid": True,
        "output_record_count": len(output),
        "upcoming_time_horizon_sec": UPCOMING_TIME_HORIZON_SEC,
        "minimum_upcoming_distance_m": MINIMUM_UPCOMING_DISTANCE_M,
        "maximum_upcoming_distance_m": MAXIMUM_UPCOMING_DISTANCE_M,
        "direction_geometry_minimum_deg": DIRECTION_GEOMETRY_MINIMUM_DEG,
        "road_level_intersection_direction_threshold_deg": ROAD_LEVEL_INTERSECTION_DIRECTION_THRESHOLD_DEG,
        "action_counts": dict(action_counts),
        "quality_status_counts": dict(quality_counts),
        "decision_source_counts": dict(source_counts),
        "text_counts": dict(text_counts),
        "unknown_reason_counts": dict(unknown_reason_counts),
        "output_sha256": sha256,
        "leakage_controls": {
            "future_ego_trajectory_used": False,
            "future_speed_used": False,
            "future_control_used": False,
            "meta_action_used_for_generation": False,
        },
    }
    atomic_write(args.output, output_text, args.force)
    atomic_write(
        args.summary_output,
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        args.force,
    )

    print("Output:", args.output)
    print("Summary:", args.summary_output)
    print("SHA-256:", sha256)
    print("Records:", len(output))
    print("Actions:", dict(action_counts))
    print("Quality:", dict(quality_counts))
    print("Decision sources:", dict(source_counts))
    print("Texts:", dict(text_counts))
    print("Unknown reasons:", dict(unknown_reason_counts))
    print("Review status: reviewed_v0.1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
