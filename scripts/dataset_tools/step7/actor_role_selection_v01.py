"""Step 7H deterministic Actor role selection v01.

The selector uses only current geometry plus frozen Step 7E visibility, Step 7F
history quality, and Step 7G road evidence. History quality is descriptive and
is not a hard eligibility gate. No future or planning input is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from step7.scene_fact_schema_v01 import ACTOR_ROLE_KEYS

VEHICLE_CLASSES = frozenset({
    "automobile", "bus", "heavy_truck", "other_vehicle", "trailer",
    "train_or_tram_car",
})
MAXIMUM_LEAD_DISTANCE_M = 80.0
MAXIMUM_SIDE_DISTANCE_M = 30.0
MAXIMUM_SIDE_REAR_M = 15.0
MAXIMUM_SIDE_FRONT_M = 30.0
MAXIMUM_SIDE_LATERAL_M = 8.0
LEAD_LATERAL_LIMIT_M = 4.5
SIDE_LATERAL_DEADBAND_M = 0.5


@dataclass(frozen=True, slots=True)
class ActorRoleCandidate:
    track_id: str
    label_class: str
    geometry: Mapping[str, Any]
    visibility: Mapping[str, Any]
    history: Mapping[str, Any]
    road: Mapping[str, Any]


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["anchor_id"]), str(row["track_id"])


def join_actor_role_inputs(
    *,
    geometry_rows: Sequence[Mapping[str, Any]],
    visibility_rows: Sequence[Mapping[str, Any]],
    history_rows: Sequence[Mapping[str, Any]],
    road_rows: Sequence[Mapping[str, Any]],
) -> tuple[ActorRoleCandidate, ...]:
    """Strictly join one Anchor's four Actor-level evidence sources."""
    sources = {
        "geometry": tuple(geometry_rows),
        "visibility": tuple(visibility_rows),
        "history": tuple(history_rows),
        "road": tuple(road_rows),
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
            raise ValueError(
                f"{name} Actor identities do not close; missing={missing} extra={extra}"
            )
    candidates = []
    for key in sorted(identities):
        geometry = indexes["geometry"][key]
        labels = {
            str(geometry["label_class"]),
            str(indexes["visibility"][key]["label_class"]),
            str(indexes["history"][key]["label_class"]),
            str(indexes["road"][key]["label_class"]),
        }
        if len(labels) != 1:
            raise ValueError(f"Actor class mismatch for {key}: {sorted(labels)}")
        candidates.append(
            ActorRoleCandidate(
                track_id=key[1],
                label_class=labels.pop(),
                geometry=geometry,
                visibility=indexes["visibility"][key],
                history=indexes["history"][key],
                road=indexes["road"][key],
            )
        )
    return tuple(candidates)


def _base_eligible(candidate: ActorRoleCandidate) -> bool:
    return (
        candidate.label_class in VEHICLE_CLASSES
        and candidate.visibility["shadow_status"] == "shadow_visible"
    )


def _role_record(candidate: ActorRoleCandidate, role: str) -> dict[str, Any]:
    geometry = candidate.geometry
    road = candidate.road
    history = candidate.history
    return {
        "role": role,
        "track_id": candidate.track_id,
        "label_class": candidate.label_class,
        "relative_x_m": float(geometry["relative_x_m"]),
        "relative_y_m": float(geometry["relative_y_m"]),
        "planar_distance_m": float(geometry["planar_distance_m"]),
        "geometric_region": str(geometry["geometric_region"]),
        "ego_lane_relation": str(road["ego_lane_relation"]),
        "lane_match_status": str(road["lane_match_status"]),
        "history_status": str(history["history_status"]),
        "mean_distance_rate_mps": history["mean_distance_rate_mps"],
        "visibility_policy_status": str(candidate.visibility["shadow_status"]),
        "winning_cell_count": int(candidate.visibility["winning_cell_count"]),
    }


def _lead_candidates(candidates: Sequence[ActorRoleCandidate]):
    result = []
    for candidate in candidates:
        if not _base_eligible(candidate):
            continue
        geometry = candidate.geometry
        x = float(geometry["relative_x_m"])
        y = float(geometry["relative_y_m"])
        distance = float(geometry["planar_distance_m"])
        relation = str(candidate.road["ego_lane_relation"])
        same_lane = relation == "same"
        geometric_fallback = relation == "unknown" and abs(y) <= LEAD_LATERAL_LIMIT_M
        if (
            x > 0.5
            and distance <= MAXIMUM_LEAD_DISTANCE_M
            and (same_lane or geometric_fallback)
        ):
            result.append((0 if same_lane else 1, x, abs(y), distance, candidate.track_id, candidate))
    return result


def _side_candidates(candidates: Sequence[ActorRoleCandidate], *, side: str):
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")
    result = []
    expected_relation = f"{side}_adjacent"
    for candidate in candidates:
        if not _base_eligible(candidate):
            continue
        geometry = candidate.geometry
        x = float(geometry["relative_x_m"])
        y = float(geometry["relative_y_m"])
        distance = float(geometry["planar_distance_m"])
        signed_lateral = y if side == "left" else -y
        side_ok = SIDE_LATERAL_DEADBAND_M < signed_lateral <= MAXIMUM_SIDE_LATERAL_M
        relation = str(candidate.road["ego_lane_relation"])
        topology_match = relation == expected_relation
        geometric_fallback = relation in ("unknown", "unrelated") and side_ok
        if (
            side_ok
            and -MAXIMUM_SIDE_REAR_M <= x <= MAXIMUM_SIDE_FRONT_M
            and distance <= MAXIMUM_SIDE_DISTANCE_M
            and (topology_match or geometric_fallback)
        ):
            result.append((0 if topology_match else 1, abs(y), abs(x), distance, candidate.track_id, candidate))
    return result


def select_actor_roles(
    *, candidates: Sequence[ActorRoleCandidate]
) -> dict[str, Any]:
    """Select three mutually exclusive roles with deterministic ranking."""
    candidate_values = tuple(candidates)
    track_ids = [item.track_id for item in candidate_values]
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("role candidates must have unique track_id values")
    selected = {}
    used = set()
    specifications = (
        ("lead_vehicle", sorted(_lead_candidates(candidate_values))),
        ("left_nearby_vehicle", sorted(_side_candidates(candidate_values, side="left"))),
        ("right_nearby_vehicle", sorted(_side_candidates(candidate_values, side="right"))),
    )
    for role, ranked in specifications:
        chosen = next((item[-1] for item in ranked if item[-1].track_id not in used), None)
        if chosen is None:
            selected[role] = None
        else:
            used.add(chosen.track_id)
            selected[role] = _role_record(chosen, role)
    if tuple(selected) != ACTOR_ROLE_KEYS:
        raise RuntimeError("role output order does not match shared schema")
    return selected


def empty_role_reasons(
    *, candidates: Sequence[ActorRoleCandidate], roles: Mapping[str, Any]
) -> dict[str, str | None]:
    """Return one compact reason for each unfilled role."""
    eligible = tuple(item for item in candidates if _base_eligible(item))
    reasons = {}
    for role in ACTOR_ROLE_KEYS:
        if roles[role] is not None:
            reasons[role] = None
        elif not candidates:
            reasons[role] = "no_current_actors"
        elif not eligible:
            reasons[role] = "no_visible_vehicle_candidates"
        else:
            reasons[role] = "no_candidate_in_role_region"
    return reasons
