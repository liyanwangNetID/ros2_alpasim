"""Join selected Step 7E policy cases with v02 and projection evidence.

The join is deterministic and read-only. The review camera is the single
occlusion-winning camera reported by v02 evidence. Projection geometry is then
resolved by Anchor, Track, and camera.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["anchor_id"]), str(row["track_id"])


def projection_identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["anchor_id"]),
        str(row["track_id"]),
        str(row["camera_name"]),
    )


def prepare_policy_visual_review_cases(
    *,
    selection: Mapping[str, Any],
    v02_rows: Sequence[Mapping[str, Any]],
    projection_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join selected cases to one winning camera and its projection row."""
    if selection.get("schema_version") != (
        "step7e-observability-policy-review-selection-v01"
    ):
        raise ValueError("unexpected review selection schema")

    selected = tuple(selection.get("cases", ()))
    selected_ids = [identity(row) for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected Anchor/Actor identities must be unique")

    v02_by_id: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in v02_rows:
        if row.get("schema_version") != "step7e-geometric-occlusion-evidence-v02":
            raise ValueError("unexpected v02 evidence schema")
        key = identity(row)
        if key in v02_by_id:
            raise ValueError("v02 Anchor/Actor identities must be unique")
        v02_by_id[key] = row

    projection_by_id: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in projection_rows:
        key = projection_identity(row)
        if key in projection_by_id:
            raise ValueError("projection Anchor/Actor/camera identities must be unique")
        projection_by_id[key] = row

    failures: Counter[str] = Counter()
    prepared: list[dict[str, Any]] = []
    for selected_row in selected:
        actor_id = identity(selected_row)
        v02 = v02_by_id.get(actor_id)
        if v02 is None:
            failures["missing_v02_row"] += 1
            continue
        if str(v02.get("label_class")) != str(selected_row.get("label_class")):
            failures["actor_class_mismatch"] += 1
            continue
        winning_cameras = tuple(
            str(value) for value in v02.get("occlusion_winning_camera_names", ())
        )
        if len(winning_cameras) != 1:
            failures["winning_camera_count_not_one"] += 1
            continue
        camera_name = winning_cameras[0]
        projection = projection_by_id.get((*actor_id, camera_name))
        if projection is None:
            failures["missing_projection_row"] += 1
            continue
        if str(projection.get("label_class")) != str(selected_row.get("label_class")):
            failures["projection_actor_class_mismatch"] += 1
            continue
        if int(projection["anchor_ns"]) != int(v02["anchor_ns"]):
            failures["anchor_timestamp_mismatch"] += 1
            continue

        joined = dict(selected_row)
        joined.update(
            {
                "anchor_ns": int(v02["anchor_ns"]),
                "camera_name": camera_name,
                "camera_selection_basis": "single_occlusion_winning_camera",
                "occluding_actor_ids": [
                    str(value) for value in v02.get("occluding_actor_ids", ())
                ],
                "total_occupied_cell_count": int(v02["total_occupied_cell_count"]),
                "total_occluded_cell_count": int(v02["total_occluded_cell_count"]),
                "clipped_bbox": projection["clipped_bbox"],
                "clipped_hull": projection["clipped_hull"],
                "inside_image_hull_area_px": float(
                    projection["inside_image_hull_area_px"]
                ),
                "inside_image_hull_ratio": float(
                    projection["inside_image_hull_ratio"]
                ),
                "projected_height_px": float(projection["projected_height_px"]),
                "minimum_depth_m": float(projection["minimum_depth_m"]),
            }
        )
        prepared.append(joined)

    return {
        "schema_version": "step7e-observability-policy-visual-review-input-v01",
        "selected_case_count": len(selected),
        "prepared_case_count": len(prepared),
        "failure_count": sum(failures.values()),
        "failure_counts": dict(sorted(failures.items())),
        "cases": prepared,
    }
