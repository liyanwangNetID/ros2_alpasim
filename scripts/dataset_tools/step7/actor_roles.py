"""Step 7H deterministic bounded Actor-list selection."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

TRAFFIC_ACTOR_CLASSES = frozenset({
    "automobile", "bus", "heavy_truck", "other_vehicle", "trailer",
    "train_or_tram_car", "rider", "person",
})
ACTOR_LIST_KEYS = ("lead_actors", "left_nearby_actors", "right_nearby_actors")
ACTOR_LIST_LIMITS = {"lead_actors": 4, "left_nearby_actors": 6, "right_nearby_actors": 6}
LEAD_FORWARD_TIME_HORIZON_SEC = 10.0
SIDE_FORWARD_TIME_HORIZON_SEC = 5.0
REAR_TIME_HORIZON_SEC = 2.0
MINIMUM_FORWARD_HORIZON_M = 20.0
MAXIMUM_SIDE_FORWARD_HORIZON_M = 120.0
MINIMUM_REAR_HORIZON_M = 15.0
MAXIMUM_REAR_HORIZON_M = 50.0
LEAD_LATERAL_LIMIT_M = 2.5
SIDE_LATERAL_DEADBAND_M = 0.5
NEARBY_LATERAL_LIMIT_M = 12.0
MAXIMUM_PERSON_SIDE_FORWARD_HORIZON_M = 30.0
NEAR_STATIONARY_SPEED_MPS = 0.5


@dataclass(frozen=True, slots=True)
class ActorRoleCandidate:
    track_id: str
    label_class: str
    geometry: Mapping[str, Any]
    visibility: Mapping[str, Any]
    history: Mapping[str, Any]
    road: Mapping[str, Any]


def _identity(row):
    return str(row["anchor_id"]), str(row["track_id"])


def join_actor_role_inputs(*, geometry_rows, visibility_rows, history_rows, road_rows):
    sources = {
        "geometry": tuple(geometry_rows), "visibility": tuple(visibility_rows),
        "history": tuple(history_rows), "road": tuple(road_rows),
    }
    indexes = {}
    for name, rows in sources.items():
        index = {}
        for row in rows:
            key = _identity(row)
            if key in index:
                raise ValueError(f"duplicate {name} Actor identity: {key}")
            index[key] = row
        indexes[name] = index
    identities = set(indexes["geometry"])
    for name in ("visibility", "history", "road"):
        if set(indexes[name]) != identities:
            missing = sorted(identities - set(indexes[name]))[:5]
            extra = sorted(set(indexes[name]) - identities)[:5]
            raise ValueError(f"{name} Actor identities do not close; missing={missing} extra={extra}")
    result = []
    for key in sorted(identities):
        labels = {str(indexes[name][key]["label_class"]) for name in indexes}
        if len(labels) != 1:
            raise ValueError(f"Actor class mismatch for {key}: {sorted(labels)}")
        result.append(ActorRoleCandidate(
            track_id=key[1], label_class=labels.pop(), geometry=indexes["geometry"][key],
            visibility=indexes["visibility"][key], history=indexes["history"][key],
            road=indexes["road"][key],
        ))
    return tuple(result)


def _base_eligible(candidate):
    return candidate.label_class in TRAFFIC_ACTOR_CLASSES and candidate.visibility["shadow_status"] == "shadow_visible"


def _role_record(candidate, role, rank):
    geometry, road, history = candidate.geometry, candidate.road, candidate.history
    return {
        "role": role, "role_rank": rank, "track_id": candidate.track_id,
        "label_class": candidate.label_class,
        "relative_x_m": float(geometry["relative_x_m"]),
        "relative_y_m": float(geometry["relative_y_m"]),
        "planar_distance_m": float(geometry["planar_distance_m"]),
        "geometric_region": str(geometry["geometric_region"]),
        "ego_lane_relation": str(road["ego_lane_relation"]),
        "lane_match_status": str(road["lane_match_status"]),
        "lane_direction_relation": str(road.get("lane_direction_relation", "unknown")),
        "history_status": str(history["history_status"]),
        "mean_distance_rate_mps": history["mean_distance_rate_mps"],
        "visibility_policy_status": str(candidate.visibility["shadow_status"]),
        "winning_cell_count": int(candidate.visibility["winning_cell_count"]),
        "actor_speed_mps": float(geometry["actor_speed_mps"]),
        "ego_speed_mps": float(geometry["ego_speed_mps"]),
        "relative_longitudinal_speed_mps": float(geometry["relative_longitudinal_speed_mps"]),
        "ego_actor_speed_gap_mps": float(geometry["ego_speed_mps"]) - float(geometry["actor_speed_mps"]),
    }


def actor_selection_horizons(reference_ego_speed_mps):
    speed = max(0.0, float(reference_ego_speed_mps))
    return {
        "reference_ego_speed_mps": speed,
        "forward_horizon_m": max(MINIMUM_FORWARD_HORIZON_M, speed * LEAD_FORWARD_TIME_HORIZON_SEC),
        "side_forward_horizon_m": min(
            MAXIMUM_SIDE_FORWARD_HORIZON_M,
            max(MINIMUM_FORWARD_HORIZON_M, speed * SIDE_FORWARD_TIME_HORIZON_SEC),
        ),
        "rear_horizon_m": min(
            MAXIMUM_REAR_HORIZON_M,
            max(MINIMUM_REAR_HORIZON_M, speed * REAR_TIME_HORIZON_SEC),
        ),
    }


def _side_forward_limit(candidate, limit):
    return min(float(limit), MAXIMUM_PERSON_SIDE_FORWARD_HORIZON_M) if candidate.label_class == "person" else float(limit)


def _side_priority(candidate):
    current = float(candidate.geometry["actor_speed_mps"])
    historical = candidate.history.get("mean_actor_speed_mps")
    if current > NEAR_STATIONARY_SPEED_MPS:
        return 0
    if historical is not None and float(historical) > NEAR_STATIONARY_SPEED_MPS:
        return 1
    return 2


def select_actor_roles(*, candidates, reference_ego_speed_mps=None):
    values = tuple(candidates)
    ids = [value.track_id for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError("role candidates must have unique track_id values")
    if reference_ego_speed_mps is None:
        reference_ego_speed_mps = max((float(value.geometry["ego_speed_mps"]) for value in values), default=0.0)
    context = actor_selection_horizons(reference_ego_speed_mps)
    forward, side_forward, rear = context["forward_horizon_m"], context["side_forward_horizon_m"], context["rear_horizon_m"]
    ranked = {key: [] for key in ACTOR_LIST_KEYS}
    for candidate in values:
        if not _base_eligible(candidate):
            continue
        geometry = candidate.geometry
        x, y = float(geometry["relative_x_m"]), float(geometry["relative_y_m"])
        if x < -rear:
            continue
        lateral, distance = abs(y), float(geometry["planar_distance_m"])
        relation = str(candidate.road["ego_lane_relation"])
        if 0.5 < x <= forward and lateral <= LEAD_LATERAL_LIMIT_M:
            closing = max(0.0, -float(geometry["relative_longitudinal_speed_mps"]))
            ttc = x / closing if closing > 1e-6 else float("inf")
            relation_priority = {"same": 0, "successor": 1, "predecessor": 1, "unknown": 2, "left_adjacent": 3, "right_adjacent": 3, "unrelated": 4}.get(relation, 5)
            ranked["lead_actors"].append((x, lateral, relation_priority, ttc, distance, candidate.track_id, candidate))
        elif -rear <= x <= _side_forward_limit(candidate, side_forward) and SIDE_LATERAL_DEADBAND_M < y <= NEARBY_LATERAL_LIMIT_M:
            ranked["left_nearby_actors"].append((_side_priority(candidate), abs(x), lateral, distance, candidate.track_id, candidate))
        elif -rear <= x <= _side_forward_limit(candidate, side_forward) and SIDE_LATERAL_DEADBAND_M < -y <= NEARBY_LATERAL_LIMIT_M:
            ranked["right_nearby_actors"].append((_side_priority(candidate), abs(x), lateral, distance, candidate.track_id, candidate))
    role_names = {"lead_actors": "lead_actor", "left_nearby_actors": "left_nearby_actor", "right_nearby_actors": "right_nearby_actor"}
    selected, metadata, used = {}, {}, set()
    for key in ACTOR_LIST_KEYS:
        ordered, chosen = sorted(ranked[key]), []
        for value in ordered:
            candidate = value[-1]
            if candidate.track_id in used:
                continue
            chosen.append(candidate)
            used.add(candidate.track_id)
            if len(chosen) == ACTOR_LIST_LIMITS[key]:
                break
        selected[key] = [_role_record(candidate, role_names[key], rank) for rank, candidate in enumerate(chosen, 1)]
        metadata[key] = {
            "candidate_count": len(ordered), "selected_count": len(chosen),
            "maximum_count": ACTOR_LIST_LIMITS[key], "truncated": len(ordered) > len(chosen),
        }
    return {**selected, "selection_context": {**context, "lists": metadata}}


def summarize_actor_role_rows(*, keyframes, rows):
    anchors = [str(row["anchor_id"]) for row in keyframes]
    if len(anchors) != len(set(anchors)) or {str(row["anchor_id"]) for row in rows} != set(anchors):
        raise ValueError("role rows must contain exactly one row per Keyframe")
    counts = Counter({key: 0 for key in ACTOR_LIST_KEYS})
    truncations = Counter({key: 0 for key in ACTOR_LIST_KEYS})
    assignments = 0
    for row in rows:
        selected_ids = []
        for key in ACTOR_LIST_KEYS:
            actors = row["roles"][key]
            counts[key] += len(actors)
            assignments += len(actors)
            truncations[key] += int(row["roles"]["selection_context"]["lists"][key]["truncated"])
            selected_ids.extend(str(actor["track_id"]) for actor in actors)
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("one Actor occupies multiple role lists in one Keyframe")
    return {
        "schema_version": "step7h-actor-role-selection-summary-v02",
        "keyframe_count": len(anchors), "role_row_count": len(rows),
        "selected_actor_counts": dict(counts), "truncated_anchor_counts": dict(truncations),
        "selected_actor_role_assignment_count": assignments, "role_conflict_count": 0,
    }
