#!/usr/bin/env python3
"""Validate all-Actor real occlusion evidence for one Step 7E camera case.

The diagnostic builds a surface raster for every Actor in one fixed real
Keyframe, resolves one shared Z-buffer, adapts all results into threshold-free
per-camera occlusion evidence, and independently verifies every reported
occluding Actor ID against actual losing competition cells.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from actor_box_projection_v01 import actor_box_corners_in_rig
from actor_camera_surface_depth_raster_v01 import (
    build_actor_camera_surface_depth_raster,
)
from actor_depth_zbuffer_v01 import (
    ActorDepthRasterInput,
    resolve_actor_depth_zbuffer,
)
from actor_occlusion_evidence_adapter_v01 import (
    build_camera_occlusion_evidence_from_zbuffer,
)
from camera_facing_box_surfaces_v01 import (
    prepare_camera_facing_box_triangles,
)
from camera_projection_v01 import load_camera_calibration
from clip_reader import DrivingClipReader
from profile_step7e_multicase_raster_stability_v01 import (
    actor_id,
    exact_inputs,
    split_anchor_id,
)
from project_paths import ALPASIM_DATA_ROOT


ANCHOR_ID = "test_clip_001_9306612661000"
CAMERA_NAME = "front_tele"
RASTER_WIDTH = 320
RASTER_HEIGHT = 180
NEAR_PLANE_M = 1e-3
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
DEPTH_TOLERANCE_M = 1e-9


def build_inputs(*, actors, ego, calibration, image_width, image_height):
    inputs = []
    unresolved_by_actor = {}

    for actor in sorted(actors, key=actor_id):
        track_id = actor_id(actor)
        corners_rig = actor_box_corners_in_rig(
            actor,
            recorded_ego_message=ego.message,
        )
        corners_camera = tuple(
            calibration.rig_point_to_camera(point)
            for point in corners_rig
        )
        prepared = prepare_camera_facing_box_triangles(
            corners_camera,
            near_plane_m=NEAR_PLANE_M,
        )
        result = build_actor_camera_surface_depth_raster(
            tuple(item.vertices_camera for item in prepared),
            calibration,
            max_angle_rad=calibration.max_angle_rad,
            maximum_depth=MAXIMUM_DEPTH,
            maximum_boundary_extent_px=MAXIMUM_BOUNDARY_EXTENT_PX,
            image_width_px=image_width,
            image_height_px=image_height,
            raster_width=RASTER_WIDTH,
            raster_height=RASTER_HEIGHT,
            near_plane_m=NEAR_PLANE_M,
            depth_tolerance_m=DEPTH_TOLERANCE_M,
        )
        if result.unresolved_boundary_count:
            unresolved_by_actor[track_id] = result.unresolved_boundary_count
        inputs.append(
            ActorDepthRasterInput(
                track_id,
                result.surface_raster,
            )
        )

    if unresolved_by_actor:
        raise RuntimeError(
            f"Unexpected unresolved Actor boundaries: {unresolved_by_actor}"
        )
    return tuple(inputs)


def independently_expected_occluders(inputs, zbuffer):
    winner_by_cell = {
        winner.cell: winner.actor_id
        for winner in zbuffer.cell_winners
    }
    expected = defaultdict(set)
    losing_cells = Counter()

    for actor_input in inputs:
        for sample in actor_input.surface_raster.cell_depths:
            winner_id = winner_by_cell.get(sample.cell)
            if winner_id is None:
                raise RuntimeError(
                    f"Missing winner for {actor_input.actor_id} cell {sample.cell}"
                )
            if winner_id != actor_input.actor_id:
                expected[actor_input.actor_id].add(winner_id)
                losing_cells[(actor_input.actor_id, winner_id)] += 1

    return {
        actor_input.actor_id: tuple(sorted(expected[actor_input.actor_id]))
        for actor_input in inputs
    }, losing_cells


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

    inputs = build_inputs(
        actors=actors,
        ego=ego,
        calibration=calibration,
        image_width=frame.value.width,
        image_height=frame.value.height,
    )
    zbuffer = resolve_actor_depth_zbuffer(
        inputs,
        depth_tolerance_m=DEPTH_TOLERANCE_M,
    )
    evidence = build_camera_occlusion_evidence_from_zbuffer(
        camera_name=CAMERA_NAME,
        raster_width=RASTER_WIDTH,
        raster_height=RASTER_HEIGHT,
        actor_rasters=inputs,
        zbuffer=zbuffer,
        static_occlusion_evaluated=False,
    )

    if len(evidence) != len(inputs) or len(evidence) != len(actors):
        raise RuntimeError("Evidence count does not match input Actor count")
    if tuple(item.track_id for item in evidence) != tuple(
        sorted(item.actor_id for item in inputs)
    ):
        raise RuntimeError("Evidence output order is not deterministic")

    expected_occluders, losing_cells = independently_expected_occluders(
        inputs,
        zbuffer,
    )

    for item in evidence:
        if item.occluding_actor_ids != expected_occluders[item.track_id]:
            raise RuntimeError(
                f"Occluder mismatch for track {item.track_id}: "
                f"actual={item.occluding_actor_ids}, "
                f"expected={expected_occluders[item.track_id]}"
            )
        if item.camera_name != CAMERA_NAME:
            raise RuntimeError(f"Camera mismatch for track {item.track_id}")
        if item.raster_width != RASTER_WIDTH or item.raster_height != RASTER_HEIGHT:
            raise RuntimeError(f"Raster mismatch for track {item.track_id}")
        if not item.actor_to_actor_occlusion_evaluated:
            raise RuntimeError(f"Occlusion not evaluated for track {item.track_id}")
        if item.static_occlusion_evaluated:
            raise RuntimeError(
                f"Static occlusion incorrectly evaluated for track {item.track_id}"
            )
        supported_losing_count = sum(
            losing_cells[(item.track_id, occluder_id)]
            for occluder_id in item.occluding_actor_ids
        )
        if supported_losing_count != item.occluded_cell_count:
            raise RuntimeError(
                f"Occluded cell support mismatch for track {item.track_id}"
            )

    evaluated = [item for item in evidence if item.evidence_status == "evaluated"]
    no_surface = [
        item for item in evidence if item.evidence_status == "no_sampled_surface"
    ]
    fully_visible = [item for item in evaluated if item.occluded_cell_count == 0]
    fully_occluded = [item for item in evaluated if item.winning_cell_count == 0]
    partially_occluded = [
        item
        for item in evaluated
        if item.winning_cell_count > 0 and item.occluded_cell_count > 0
    ]
    actors_with_occluders = [item for item in evidence if item.occluding_actor_ids]

    if len(evaluated) + len(no_surface) != len(evidence):
        raise RuntimeError("Unexpected evidence status")
    if len(fully_visible) + len(fully_occluded) + len(partially_occluded) != len(evaluated):
        raise RuntimeError("Evaluated evidence categories do not close")

    pair_counts = Counter()
    for (target_id, occluder_id), count in losing_cells.items():
        pair_counts[(target_id, occluder_id)] += count

    print("All real Actor occlusion-evidence adapter check")
    print("anchor:", ANCHOR_ID)
    print("camera:", CAMERA_NAME)
    print("raster:", f"{RASTER_WIDTH}x{RASTER_HEIGHT}")
    print("input_actor_count:", len(actors))
    print("evaluated_actor_count:", len(evaluated))
    print("no_sampled_surface_actor_count:", len(no_surface))
    print("fully_visible_actor_count:", len(fully_visible))
    print("partially_occluded_actor_count:", len(partially_occluded))
    print("fully_occluded_actor_count:", len(fully_occluded))
    print("actors_with_occluders_count:", len(actors_with_occluders))
    print("unique_target_occluder_pair_count:", len(pair_counts))
    print("occupied_union_cell_count:", zbuffer.occupied_union_cell_count)
    print("contested_cell_count:", zbuffer.contested_cell_count)
    print("top target-occluder pairs by losing cells:")
    for (target_id, occluder_id), count in sorted(
        pair_counts.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )[:20]:
        print(
            f"  target={target_id} occluder={occluder_id} "
            f"losing_cells={count}"
        )

    print(
        "PASS: all real Actor occlusion evidence is supported by actual "
        "losing Z-buffer competition cells."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
