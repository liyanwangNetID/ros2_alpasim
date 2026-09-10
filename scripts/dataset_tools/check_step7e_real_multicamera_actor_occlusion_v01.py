#!/usr/bin/env python3
"""Validate the real four-camera Actor occlusion composition for Step 7E.

This diagnostic runs the reusable multicamera geometry-to-occlusion pipeline
at one fixed Keyframe. Each camera resolves an independent shared Z-buffer;
per-camera evidence is then grouped by Actor without selecting final
observability labels or evaluating static-scene occlusion.
"""

from __future__ import annotations

from collections import Counter

from camera_projection_v01 import load_camera_calibration
from clip_reader import DrivingClipReader
from multicamera_actor_occlusion_from_geometry_v01 import (
    CameraOcclusionBuildInput,
    build_multicamera_actor_occlusion_from_geometry,
)
from profile_step7e_multicase_raster_stability_v01 import (
    exact_inputs,
    split_anchor_id,
)
from project_paths import ALPASIM_DATA_ROOT
from scene_fact_schema_v01 import CAMERA_NAMES


ANCHOR_ID = "test_clip_001_9306612661000"
RASTER_WIDTH = 480
RASTER_HEIGHT = 270
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
NEAR_PLANE_M = 1e-3
DEPTH_TOLERANCE_M = 1e-9


def load_camera_inputs(reader, anchor_ns):
    result = {}
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
        result[camera_name] = CameraOcclusionBuildInput(
            calibration=calibration,
            image_width_px=frame.value.width,
            image_height_px=frame.value.height,
        )
    return result


def main() -> int:
    clip_id, anchor_ns = split_anchor_id(ANCHOR_ID)
    reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
    ego, actors = exact_inputs(reader, anchor_ns)
    cameras = load_camera_inputs(reader, anchor_ns)

    result = build_multicamera_actor_occlusion_from_geometry(
        actors=actors,
        recorded_ego_message=ego.message,
        cameras=cameras,
        raster_width=RASTER_WIDTH,
        raster_height=RASTER_HEIGHT,
        maximum_depth=MAXIMUM_DEPTH,
        maximum_boundary_extent_px=MAXIMUM_BOUNDARY_EXTENT_PX,
        near_plane_m=NEAR_PLANE_M,
        depth_tolerance_m=DEPTH_TOLERANCE_M,
    )

    if result.actor_count != len(actors):
        raise RuntimeError("Multicamera Actor count does not match input")
    if len(result.camera_results) != len(CAMERA_NAMES):
        raise RuntimeError("Camera result count does not match CAMERA_NAMES")
    if len(result.actor_evidence) != len(actors):
        raise RuntimeError("Grouped Actor evidence count does not match input")
    if len(result.actor_summaries) != len(actors):
        raise RuntimeError("Actor summary count does not match input")

    ordered_ids = tuple(item.track_id for item in result.actor_evidence)
    if ordered_ids != tuple(sorted(ordered_ids)):
        raise RuntimeError("Grouped Actor evidence is not deterministically sorted")
    summary_ids = tuple(item.track_id for item in result.actor_summaries)
    if summary_ids != ordered_ids:
        raise RuntimeError("Actor summaries and grouped evidence IDs differ")

    total_unresolved = 0
    camera_summaries = []
    camera_evidence_by_name = {}

    for camera_result in result.camera_results:
        camera_name = camera_result.camera_name
        evidence = camera_result.occlusion.actor_evidence
        camera_evidence_by_name[camera_name] = {
            item.track_id: item for item in evidence
        }
        if len(evidence) != len(actors):
            raise RuntimeError(
                f"Evidence count mismatch for camera {camera_name}"
            )

        evaluated = [
            item for item in evidence if item.evidence_status == "evaluated"
        ]
        no_surface = [
            item
            for item in evidence
            if item.evidence_status == "no_sampled_surface"
        ]
        unexpected = [
            item
            for item in evidence
            if item.evidence_status not in {"evaluated", "no_sampled_surface"}
        ]
        if unexpected:
            raise RuntimeError(
                f"Unexpected evidence status in camera {camera_name}"
            )
        if len(evaluated) + len(no_surface) != len(actors):
            raise RuntimeError(
                f"Evidence status counts do not close for camera {camera_name}"
            )

        fully_visible = sum(
            item.occluded_cell_count == 0 for item in evaluated
        )
        fully_occluded = sum(
            item.winning_cell_count == 0 for item in evaluated
        )
        partially_occluded = sum(
            item.winning_cell_count > 0 and item.occluded_cell_count > 0
            for item in evaluated
        )
        if fully_visible + fully_occluded + partially_occluded != len(evaluated):
            raise RuntimeError(
                f"Visibility count categories do not close for {camera_name}"
            )

        unresolved = sum(
            item.unresolved_boundary_count
            for item in camera_result.surface_diagnostics
        )
        total_unresolved += unresolved
        if unresolved:
            raise RuntimeError(
                f"Unexpected unresolved boundary count in {camera_name}: "
                f"{unresolved}"
            )

        zbuffer = camera_result.occlusion.zbuffer
        camera_summaries.append(
            {
                "camera_name": camera_name,
                "evaluated": len(evaluated),
                "no_surface": len(no_surface),
                "fully_visible": fully_visible,
                "partially_occluded": partially_occluded,
                "fully_occluded": fully_occluded,
                "union": zbuffer.occupied_union_cell_count,
                "contested": zbuffer.contested_cell_count,
            }
        )

    camera_order = tuple(item["camera_name"] for item in camera_summaries)
    if camera_order != tuple(CAMERA_NAMES):
        raise RuntimeError("Real camera results are not in canonical order")

    multicamera_status_counts = Counter()
    actors_with_any_evaluated_camera = 0
    actors_with_any_winning_camera = 0
    actors_with_any_occluder = 0

    for grouped in result.actor_evidence:
        grouped_camera_order = tuple(
            item.camera_name for item in grouped.camera_evidence
        )
        if grouped_camera_order != tuple(CAMERA_NAMES):
            raise RuntimeError(
                f"Camera order mismatch for track {grouped.track_id}"
            )
        for item in grouped.camera_evidence:
            expected = camera_evidence_by_name[item.camera_name][grouped.track_id]
            if item != expected:
                raise RuntimeError(
                    f"Grouped evidence mismatch for track {grouped.track_id} "
                    f"camera {item.camera_name}"
                )

        evaluated = [
            item
            for item in grouped.camera_evidence
            if item.evidence_status == "evaluated"
        ]
        if evaluated:
            actors_with_any_evaluated_camera += 1
        if any(item.winning_cell_count > 0 for item in evaluated):
            actors_with_any_winning_camera += 1
        if any(item.occluding_actor_ids for item in evaluated):
            actors_with_any_occluder += 1

        evaluated_count = len(evaluated)
        winning_camera_count = sum(
            item.winning_cell_count > 0 for item in evaluated
        )
        multicamera_status_counts[
            (evaluated_count, winning_camera_count)
        ] += 1

    grouped_by_id = {
        item.track_id: item for item in result.actor_evidence
    }
    summary_status_counts = Counter()
    for summary in result.actor_summaries:
        grouped = grouped_by_id[summary.track_id]
        evaluated = tuple(
            item
            for item in grouped.camera_evidence
            if item.evidence_status == "evaluated"
        )
        expected_evaluated_names = tuple(
            item.camera_name for item in evaluated
        )
        expected_winning_names = tuple(
            item.camera_name
            for item in evaluated
            if item.winning_cell_count > 0
        )
        expected_occluders = tuple(sorted({
            actor_id
            for item in evaluated
            for actor_id in item.occluding_actor_ids
        }))
        expected_maximum_fraction = (
            max(item.visible_fraction for item in evaluated)
            if evaluated
            else None
        )
        if summary.evaluated_camera_names != expected_evaluated_names:
            raise RuntimeError(
                f"Evaluated camera summary mismatch for track {summary.track_id}"
            )
        if summary.winning_camera_names != expected_winning_names:
            raise RuntimeError(
                f"Winning camera summary mismatch for track {summary.track_id}"
            )
        if summary.occluding_actor_ids != expected_occluders:
            raise RuntimeError(
                f"Occluder summary mismatch for track {summary.track_id}"
            )
        if summary.maximum_visible_fraction != expected_maximum_fraction:
            raise RuntimeError(
                f"Maximum visible fraction mismatch for track {summary.track_id}"
            )
        if summary.total_occupied_cell_count != sum(
            item.occupied_cell_count for item in evaluated
        ):
            raise RuntimeError(
                f"Total occupied count mismatch for track {summary.track_id}"
            )
        if summary.total_winning_cell_count != sum(
            item.winning_cell_count for item in evaluated
        ):
            raise RuntimeError(
                f"Total winning count mismatch for track {summary.track_id}"
            )
        if summary.total_occluded_cell_count != sum(
            item.occluded_cell_count for item in evaluated
        ):
            raise RuntimeError(
                f"Total occluded count mismatch for track {summary.track_id}"
            )
        if not summary.actor_to_actor_occlusion_evaluated:
            raise RuntimeError(
                f"Unexpected unevaluated summary for track {summary.track_id}"
            )
        if summary.static_occlusion_evaluated:
            raise RuntimeError(
                f"Static occlusion incorrectly evaluated for track {summary.track_id}"
            )
        summary_status_counts[
            (
                summary.evaluated_camera_count,
                summary.winning_camera_count,
                summary.maximum_visible_fraction is None,
            )
        ] += 1

    print("Real four-camera Actor occlusion composition check")
    print("anchor:", ANCHOR_ID)
    print("raster:", f"{RASTER_WIDTH}x{RASTER_HEIGHT}")
    print("input_actor_count:", len(actors))
    print("camera summaries:")
    for item in camera_summaries:
        print(
            f"  {item['camera_name']}: "
            f"evaluated={item['evaluated']} "
            f"no_surface={item['no_surface']} "
            f"fully_visible={item['fully_visible']} "
            f"partially_occluded={item['partially_occluded']} "
            f"fully_occluded={item['fully_occluded']} "
            f"union={item['union']} "
            f"contested={item['contested']}"
        )
    print("total_unresolved_boundary_count:", total_unresolved)
    print(
        "actors_with_any_evaluated_camera:",
        actors_with_any_evaluated_camera,
    )
    print(
        "actors_with_any_winning_camera:",
        actors_with_any_winning_camera,
    )
    print("actors_with_any_occluder:", actors_with_any_occluder)
    print("evaluated-camera / winning-camera distribution:")
    for (evaluated_count, winning_count), count in sorted(
        multicamera_status_counts.items()
    ):
        print(
            f"  evaluated_cameras={evaluated_count} "
            f"winning_cameras={winning_count} actors={count}"
        )
    print("actor summary distribution:")
    for (
        evaluated_count,
        winning_count,
        maximum_fraction_is_none,
    ), count in sorted(summary_status_counts.items()):
        print(
            f"  evaluated_cameras={evaluated_count} "
            f"winning_cameras={winning_count} "
            f"maximum_fraction_none={maximum_fraction_is_none} "
            f"actors={count}"
        )

    print(
        "PASS: real four-camera occlusion evidence is complete, ordered, "
        "and consistent with each independent camera result."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
