#!/usr/bin/env python3
"""Validate the complete real Step 7E geometric-occlusion evidence pipeline.

Runs four-camera geometric projection, four-camera Actor-to-Actor Z-buffer
occlusion, evidence joining, and the combined summary from one fixed current
Actor snapshot. No final observability label or static-scene occlusion result
is produced.
"""

from __future__ import annotations

from step7.actor_geometric_occlusion_from_geometry_v01 import (
    CameraGeometricOcclusionBuildInput,
    build_actor_geometric_occlusion_from_geometry,
)
from step7.camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from step7.profile_step7e_multicase_raster_stability_v01 import (
    exact_inputs,
    split_anchor_id,
)
from project_paths import ALPASIM_DATA_ROOT
from step7.scene_fact_schema_v01 import CAMERA_NAMES


ANCHOR_ID = "test_clip_001_9306612661000"
RASTER_WIDTH = 480
RASTER_HEIGHT = 270
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
NEAR_PLANE_M = 1e-3
MAXIMUM_CHORD_ERROR_PX = 1.0
MAXIMUM_ADAPTIVE_DEPTH = 14
DEPTH_TOLERANCE_M = 1e-9


def load_camera_inputs(reader, anchor_ns):
    cameras = {}
    for camera_name in CAMERA_NAMES:
        frame = reader.camera_indexes[camera_name].exact(anchor_ns)
        if frame is None:
            raise RuntimeError(
                f"Exact camera frame unavailable: {ANCHOR_ID} {camera_name}"
            )
        calibration = load_camera_calibration(
            reader.clip_directory / "calibration" / f"{camera_name}.json",
            camera_name=camera_name,
            source_width=frame.value.width,
            source_height=frame.value.height,
        )
        if calibration.max_angle_rad is None:
            raise RuntimeError(
                f"Camera calibration has no max_angle_rad: {camera_name}"
            )
        cameras[camera_name] = CameraGeometricOcclusionBuildInput(
            calibration=calibration,
            image_width_px=frame.value.width,
            image_height_px=frame.value.height,
        )
    return cameras


def main() -> int:
    clip_id, anchor_ns = split_anchor_id(ANCHOR_ID)
    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
    ego, actors = exact_inputs(reader, anchor_ns)
    cameras = load_camera_inputs(reader, anchor_ns)

    result = build_actor_geometric_occlusion_from_geometry(
        actors=actors,
        recorded_ego_message=ego.message,
        cameras=cameras,
        raster_width=RASTER_WIDTH,
        raster_height=RASTER_HEIGHT,
        maximum_depth=MAXIMUM_DEPTH,
        maximum_boundary_extent_px=MAXIMUM_BOUNDARY_EXTENT_PX,
        samples_per_edge=None,
        near_plane_m=NEAR_PLANE_M,
        maximum_chord_error_px=MAXIMUM_CHORD_ERROR_PX,
        maximum_adaptive_depth=MAXIMUM_ADAPTIVE_DEPTH,
        depth_tolerance_m=DEPTH_TOLERANCE_M,
    )

    if result.actor_count != len(actors):
        raise RuntimeError("Complete pipeline Actor count differs from input")

    geometric_ids = tuple(
        item.track_id for item in result.geometric.actor_observability
    )
    occlusion_ids = tuple(
        item.track_id for item in result.occlusion.actor_summaries
    )
    combined = result.combined.combined_evidence.actor_evidence
    combined_ids = tuple(item.track_id for item in combined)
    if geometric_ids != occlusion_ids or geometric_ids != combined_ids:
        raise RuntimeError("Complete pipeline Actor ID sequences differ")

    summary = result.combined.combined_summary
    if summary.actor_count != len(actors):
        raise RuntimeError("Combined summary Actor count differs from input")
    if summary.actor_to_actor_occlusion_complete_count != len(actors):
        raise RuntimeError("Actor-to-Actor occlusion is not complete")
    if summary.static_occlusion_evaluated_count != 0:
        raise RuntimeError("Static occlusion was unexpectedly evaluated")
    if summary.reasons:
        raise RuntimeError(f"Unexpected combined-summary reasons: {summary.reasons}")

    unresolved_by_camera = {}
    for camera_result in result.occlusion.camera_results:
        unresolved = sum(
            item.unresolved_boundary_count
            for item in camera_result.surface_diagnostics
        )
        unresolved_by_camera[camera_result.camera_name] = unresolved
        if unresolved:
            raise RuntimeError(
                f"Unexpected unresolved boundaries in "
                f"{camera_result.camera_name}: {unresolved}"
            )

    geometric_status_counts = {}
    for item in result.geometric.actor_observability:
        geometric_status_counts[item.observability_status] = (
            geometric_status_counts.get(item.observability_status, 0) + 1
        )

    combined_by_id = {item.track_id: item for item in combined}
    track_13 = combined_by_id["13"]
    track_29 = combined_by_id["29"]
    if "29" not in track_13.occluding_actor_ids:
        raise RuntimeError("Track 13 no longer records track 29 as an occluder")
    if track_29.total_winning_cell_count <= 0:
        raise RuntimeError("Track 29 has no winning cells across cameras")

    print("Real complete geometric-occlusion pipeline check")
    print("anchor:", ANCHOR_ID)
    print("raster:", f"{RASTER_WIDTH}x{RASTER_HEIGHT}")
    print("input_actor_count:", len(actors))
    print("geometric_status_counts:", dict(sorted(geometric_status_counts.items())))
    print("combined_evidence_status_counts:", dict(summary.evidence_status_counts))
    print(
        "geometric_candidate_actor_count:",
        summary.geometric_candidate_actor_count,
    )
    print(
        "geometric_candidate_with_sampled_surface_actor_count:",
        summary.geometric_candidate_with_sampled_surface_actor_count,
    )
    print(
        "geometric_candidate_with_winning_cells_actor_count:",
        summary.geometric_candidate_with_winning_cells_actor_count,
    )
    print(
        "geometric_candidate_without_sampled_surface_actor_count:",
        summary.geometric_candidate_without_sampled_surface_actor_count,
    )
    print(
        "geometric_candidate_fully_occluded_actor_count:",
        summary.geometric_candidate_fully_occluded_actor_count,
    )
    print("actor_with_occluder_count:", summary.actor_with_occluder_count)
    print(
        "actor_to_actor_occlusion_complete_count:",
        summary.actor_to_actor_occlusion_complete_count,
    )
    print(
        "static_occlusion_evaluated_count:",
        summary.static_occlusion_evaluated_count,
    )
    print("unresolved_boundary_count_by_camera:", unresolved_by_camera)
    print(
        "candidate_without_sampled_surface_track_ids:",
        list(summary.geometric_candidate_without_sampled_surface_track_ids),
    )
    print(
        "candidate_fully_occluded_track_ids:",
        list(summary.geometric_candidate_fully_occluded_track_ids),
    )
    print("track 13 combined evidence:")
    print(
        "  geometric_candidates=",
        list(track_13.geometric_candidate_camera_names),
    )
    print(
        "  candidate_with_winners=",
        list(track_13.geometric_candidate_with_winning_cells_camera_names),
    )
    print("  occluders=", list(track_13.occluding_actor_ids))
    print("  maximum_visible_fraction=", track_13.maximum_visible_fraction)
    print("track 29 combined evidence:")
    print(
        "  geometric_candidates=",
        list(track_29.geometric_candidate_camera_names),
    )
    print(
        "  candidate_with_winners=",
        list(track_29.geometric_candidate_with_winning_cells_camera_names),
    )
    print("  occluders=", list(track_29.occluding_actor_ids))
    print("  maximum_visible_fraction=", track_29.maximum_visible_fraction)
    print(
        "PASS: complete real geometric and Actor-occlusion evidence was "
        "built and joined without final visibility thresholds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
