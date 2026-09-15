#!/usr/bin/env python3
"""Profile complete Step 7E geometric-occlusion evidence across fixed Anchors.

Runs the reusable end-to-end current-geometry pipeline for five fixed Anchors,
collects per-Anchor combined-evidence summaries, and writes deterministic JSON.
No final visibility labels, static-scene occlusion, or occlusion thresholds are
applied.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

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


ANCHOR_IDS = (
    "test_clip_001_9306612661000",
    "test_clip_063_18787721418000",
    "test_clip_564_95233721166000",
    "test_clip_316_1404486225261000",
)
RASTER_WIDTH = 480
RASTER_HEIGHT = 270
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
NEAR_PLANE_M = 1e-3
MAXIMUM_CHORD_ERROR_PX = 1.0
MAXIMUM_ADAPTIVE_DEPTH = 14
DEPTH_TOLERANCE_M = 1e-9
OUTPUT = Path("/tmp/step7e_multicase_geometric_occlusion_profile.json")


def load_camera_inputs(reader, anchor_id, anchor_ns):
    cameras = {}
    for camera_name in CAMERA_NAMES:
        frame = reader.camera_indexes[camera_name].exact(anchor_ns)
        if frame is None:
            raise RuntimeError(
                f"Exact camera frame unavailable: {anchor_id} {camera_name}"
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
    case_rows = []
    aggregate_geometric_status = Counter()
    aggregate_combined_status = Counter()
    aggregate_review_without_surface = []
    aggregate_review_fully_occluded = []

    for case_index, anchor_id in enumerate(ANCHOR_IDS, start=1):
        clip_id, anchor_ns = split_anchor_id(anchor_id)
        reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
        ego, actors = exact_inputs(reader, anchor_ns)
        cameras = load_camera_inputs(reader, anchor_id, anchor_ns)

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
            raise RuntimeError(f"Actor count mismatch: {anchor_id}")

        geometric_status = Counter(
            item.observability_status
            for item in result.geometric.actor_observability
        )
        summary = result.combined.combined_summary
        combined_status = Counter(dict(summary.evidence_status_counts))
        aggregate_geometric_status.update(geometric_status)
        aggregate_combined_status.update(combined_status)

        unresolved_by_camera = {}
        for camera_result in result.occlusion.camera_results:
            unresolved = sum(
                item.unresolved_boundary_count
                for item in camera_result.surface_diagnostics
            )
            unresolved_by_camera[camera_result.camera_name] = unresolved
            if unresolved:
                raise RuntimeError(
                    f"Unresolved boundaries: {anchor_id} "
                    f"{camera_result.camera_name} count={unresolved}"
                )

        if summary.actor_to_actor_occlusion_complete_count != len(actors):
            raise RuntimeError(f"Incomplete Actor occlusion: {anchor_id}")
        if summary.static_occlusion_evaluated_count != 0:
            raise RuntimeError(f"Unexpected static occlusion: {anchor_id}")
        if summary.reasons:
            raise RuntimeError(
                f"Unexpected combined-summary reasons: {anchor_id} "
                f"{summary.reasons}"
            )

        without_surface_keys = [
            f"{anchor_id}/{track_id}"
            for track_id in (
                summary.geometric_candidate_without_sampled_surface_track_ids
            )
        ]
        fully_occluded_keys = [
            f"{anchor_id}/{track_id}"
            for track_id in summary.geometric_candidate_fully_occluded_track_ids
        ]
        aggregate_review_without_surface.extend(without_surface_keys)
        aggregate_review_fully_occluded.extend(fully_occluded_keys)

        combined_evidence = result.combined.combined_evidence.actor_evidence
        case_row = {
            "anchor_id": anchor_id,
            "clip_id": clip_id,
            "anchor_ns": anchor_ns,
            "actor_count": len(actors),
            "geometric_status_counts": dict(sorted(geometric_status.items())),
            "combined_summary": summary.to_dict(),
            "unresolved_boundary_count_by_camera": unresolved_by_camera,
            "combined_evidence": [
                item.to_dict() for item in combined_evidence
            ],
        }
        case_rows.append(case_row)

        print(f"Case {case_index}/{len(ANCHOR_IDS)}: {anchor_id}")
        print("  actors:", len(actors))
        print("  geometric_status_counts:", dict(sorted(geometric_status.items())))
        print("  combined_status_counts:", dict(sorted(combined_status.items())))
        print(
            "  candidate_with_surface:",
            summary.geometric_candidate_with_sampled_surface_actor_count,
        )
        print(
            "  candidate_with_winners:",
            summary.geometric_candidate_with_winning_cells_actor_count,
        )
        print(
            "  candidate_without_surface:",
            summary.geometric_candidate_without_sampled_surface_actor_count,
        )
        print(
            "  candidate_fully_occluded:",
            summary.geometric_candidate_fully_occluded_actor_count,
        )
        print("  actors_with_occluders:", summary.actor_with_occluder_count)

    aggregate = {
        "case_count": len(case_rows),
        "actor_count": sum(row["actor_count"] for row in case_rows),
        "geometric_status_counts": dict(
            sorted(aggregate_geometric_status.items())
        ),
        "combined_evidence_status_counts": dict(
            sorted(aggregate_combined_status.items())
        ),
        "geometric_candidate_actor_count": sum(
            row["combined_summary"]["geometric_candidate_actor_count"]
            for row in case_rows
        ),
        "geometric_candidate_with_sampled_surface_actor_count": sum(
            row["combined_summary"][
                "geometric_candidate_with_sampled_surface_actor_count"
            ]
            for row in case_rows
        ),
        "geometric_candidate_with_winning_cells_actor_count": sum(
            row["combined_summary"][
                "geometric_candidate_with_winning_cells_actor_count"
            ]
            for row in case_rows
        ),
        "geometric_candidate_without_sampled_surface_actor_count": len(
            aggregate_review_without_surface
        ),
        "geometric_candidate_fully_occluded_actor_count": len(
            aggregate_review_fully_occluded
        ),
        "actor_with_occluder_count": sum(
            row["combined_summary"]["actor_with_occluder_count"]
            for row in case_rows
        ),
        "geometric_candidate_without_sampled_surface_actor_keys": sorted(
            aggregate_review_without_surface
        ),
        "geometric_candidate_fully_occluded_actor_keys": sorted(
            aggregate_review_fully_occluded
        ),
    }

    report = {
        "schema_version": "step7e-multicase-geometric-occlusion-profile-v01",
        "raster_width": RASTER_WIDTH,
        "raster_height": RASTER_HEIGHT,
        "maximum_depth": MAXIMUM_DEPTH,
        "maximum_boundary_extent_px": MAXIMUM_BOUNDARY_EXTENT_PX,
        "near_plane_m": NEAR_PLANE_M,
        "maximum_chord_error_px": MAXIMUM_CHORD_ERROR_PX,
        "maximum_adaptive_depth": MAXIMUM_ADAPTIVE_DEPTH,
        "depth_tolerance_m": DEPTH_TOLERANCE_M,
        "aggregate": aggregate,
        "cases": case_rows,
    }
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Aggregate")
    for key, value in aggregate.items():
        if not key.endswith("_actor_keys"):
            print(f"  {key}:", value)
    print("  geometric_candidate_without_sampled_surface_actor_keys:")
    for value in aggregate[
        "geometric_candidate_without_sampled_surface_actor_keys"
    ]:
        print("   ", value)
    print("Output:", OUTPUT)
    print(
        "PASS: multicase complete geometric-occlusion evidence profile "
        "completed without final visibility thresholds."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
