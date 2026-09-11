#!/usr/bin/env python3
"""Profile real Actor occlusion stability between 480x270 and 640x360.

This diagnostic runs the reusable current-geometry pipeline twice for one fixed
cross_left Keyframe, compares every Actor with the threshold-free resolution
comparison contract, and reports surface, winner, occluder-set, and visible-
fraction changes. It does not select observability labels or acceptance limits.
"""

from __future__ import annotations

from collections import Counter

from actor_camera_occlusion_from_geometry_v01 import (
    build_actor_camera_occlusion_from_geometry,
)
from actor_occlusion_resolution_comparison_v01 import (
    compare_actor_occlusion_resolutions,
)
from camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from profile_step7e_multicase_raster_stability_v01 import (
    exact_inputs,
    split_anchor_id,
)
from project_paths import ALPASIM_DATA_ROOT


ANCHOR_ID = "test_clip_564_95233721166000"
CAMERA_NAME = "cross_left"
CANDIDATE_RASTER = (480, 270)
REFERENCE_RASTER = (640, 360)
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
NEAR_PLANE_M = 1e-3
DEPTH_TOLERANCE_M = 1e-9
TARGET_TRACK_ID = "226"


def run_resolution(*, actors, ego, calibration, image_width, image_height, raster):
    width, height = raster
    return build_actor_camera_occlusion_from_geometry(
        actors=actors,
        recorded_ego_message=ego.message,
        calibration=calibration,
        camera_name=CAMERA_NAME,
        image_width_px=image_width,
        image_height_px=image_height,
        raster_width=width,
        raster_height=height,
        maximum_depth=MAXIMUM_DEPTH,
        maximum_boundary_extent_px=MAXIMUM_BOUNDARY_EXTENT_PX,
        near_plane_m=NEAR_PLANE_M,
        depth_tolerance_m=DEPTH_TOLERANCE_M,
    )


def main() -> int:
    clip_id, anchor_ns = split_anchor_id(ANCHOR_ID)
    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
    ego, actors = exact_inputs(reader, anchor_ns)

    frame = reader.camera_indexes[CAMERA_NAME].exact(anchor_ns)
    if frame is None:
        raise RuntimeError(
            f"Exact camera frame unavailable: {ANCHOR_ID} {CAMERA_NAME}"
        )
    calibration = load_camera_calibration(
        reader.clip_directory / "calibration" / f"{CAMERA_NAME}.json",
        camera_name=CAMERA_NAME,
        source_width=frame.value.width,
        source_height=frame.value.height,
    )
    if calibration.max_angle_rad is None:
        raise RuntimeError("Camera calibration has no max_angle_rad")

    candidate = run_resolution(
        actors=actors,
        ego=ego,
        calibration=calibration,
        image_width=frame.value.width,
        image_height=frame.value.height,
        raster=CANDIDATE_RASTER,
    )
    reference = run_resolution(
        actors=actors,
        ego=ego,
        calibration=calibration,
        image_width=frame.value.width,
        image_height=frame.value.height,
        raster=REFERENCE_RASTER,
    )

    candidate_by_id = {
        item.track_id: item
        for item in candidate.occlusion.actor_evidence
    }
    reference_by_id = {
        item.track_id: item
        for item in reference.occlusion.actor_evidence
    }
    if set(candidate_by_id) != set(reference_by_id):
        raise RuntimeError("Candidate and reference Actor sets differ")
    if len(candidate_by_id) != len(actors):
        raise RuntimeError("Comparison Actor count does not match input")

    comparisons = tuple(
        compare_actor_occlusion_resolutions(
            candidate=candidate_by_id[track_id],
            reference=reference_by_id[track_id],
        )
        for track_id in sorted(candidate_by_id)
    )

    comparable = tuple(item for item in comparisons if item.both_evaluated)
    surface_changes = tuple(
        item for item in comparisons if item.sampled_surface_presence_changed
    )
    winner_changes = tuple(
        item for item in comparable if item.winning_presence_changed
    )
    occluder_changes = tuple(
        item for item in comparable if item.occluding_actor_set_changed
    )
    nonzero_deltas = tuple(
        item
        for item in comparable
        if item.absolute_visible_fraction_delta > 0.0
    )
    ordered_deltas = tuple(
        sorted(
            nonzero_deltas,
            key=lambda item: (
                -item.absolute_visible_fraction_delta,
                item.track_id,
            ),
        )
    )

    status_pairs = Counter(
        (
            item.candidate_evidence_status,
            item.reference_evidence_status,
        )
        for item in comparisons
    )

    candidate_unresolved = sum(
        item.unresolved_boundary_count
        for item in candidate.surface_diagnostics
    )
    reference_unresolved = sum(
        item.unresolved_boundary_count
        for item in reference.surface_diagnostics
    )
    if candidate_unresolved or reference_unresolved:
        raise RuntimeError(
            "Unexpected unresolved boundary diagnostics: "
            f"candidate={candidate_unresolved}, "
            f"reference={reference_unresolved}"
        )

    target = next(
        (item for item in comparisons if item.track_id == TARGET_TRACK_ID),
        None,
    )
    if target is None:
        raise RuntimeError(f"Target track unavailable: {TARGET_TRACK_ID}")
    if not target.both_evaluated:
        raise RuntimeError("Target track is not comparable")
    if not target.winning_presence_changed:
        raise RuntimeError("Target track winning-presence baseline changed")
    if target.candidate_winning_cell_count != 0:
        raise RuntimeError("Target candidate winning count baseline changed")
    if target.reference_winning_cell_count != 4:
        raise RuntimeError("Target reference winning count baseline changed")

    print("Real Actor occlusion resolution-comparison profile")
    print("anchor:", ANCHOR_ID)
    print("camera:", CAMERA_NAME)
    print("candidate_raster:", f"{CANDIDATE_RASTER[0]}x{CANDIDATE_RASTER[1]}")
    print("reference_raster:", f"{REFERENCE_RASTER[0]}x{REFERENCE_RASTER[1]}")
    print("input_actor_count:", len(actors))
    print("comparable_actor_count:", len(comparable))
    print("sampled_surface_presence_change_count:", len(surface_changes))
    print("winning_presence_change_count:", len(winner_changes))
    print("occluding_actor_set_change_count:", len(occluder_changes))
    print("nonzero_visible_fraction_delta_count:", len(nonzero_deltas))
    print(
        "maximum_absolute_visible_fraction_delta:",
        ordered_deltas[0].absolute_visible_fraction_delta
        if ordered_deltas
        else 0.0,
    )
    print("status-pair counts:")
    for pair, count in sorted(status_pairs.items()):
        print(f"  candidate={pair[0]} reference={pair[1]} actors={count}")
    print("largest visible-fraction deltas:")
    for item in ordered_deltas[:20]:
        print(
            f"  track={item.track_id} "
            f"delta={item.absolute_visible_fraction_delta:.6f} "
            f"candidate_occupied={item.candidate_occupied_cell_count} "
            f"candidate_winning={item.candidate_winning_cell_count} "
            f"reference_occupied={item.reference_occupied_cell_count} "
            f"reference_winning={item.reference_winning_cell_count} "
            f"winning_presence_changed={item.winning_presence_changed} "
            f"occluder_set_changed={item.occluding_actor_set_changed}"
        )
    print("track 226 comparison:")
    print(
        f"  both_evaluated={target.both_evaluated} "
        f"surface_presence_changed={target.sampled_surface_presence_changed} "
        f"winning_presence_changed={target.winning_presence_changed} "
        f"candidate_winning={target.candidate_winning_cell_count} "
        f"reference_winning={target.reference_winning_cell_count} "
        f"visible_fraction_delta={target.absolute_visible_fraction_delta} "
        f"occluder_set_changed={target.occluding_actor_set_changed}"
    )
    print(
        "PASS: real resolution-comparison evidence preserves all Actor "
        "surface, winner, occluder, and fraction changes without thresholds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
