#!/usr/bin/env python3
"""Profile real Actor occlusion resolution stability across fixed cases.

This Step 7E diagnostic runs the reusable current-geometry pipeline at 480x270
and 640x360 for five fixed clip/camera cases, builds threshold-free per-Actor
resolution profiles, and writes deterministic JSON diagnostics. It does not
select observability labels or define raster acceptance thresholds.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median

from actor_camera_occlusion_from_geometry_v01 import (
    build_actor_camera_occlusion_from_geometry,
)
from camera_projection_v01 import load_camera_calibration
from clip_reader import DrivingClipReader
from occlusion_resolution_profile_v01 import (
    build_occlusion_resolution_profile,
)
from profile_step7e_multicase_raster_stability_v01 import (
    exact_inputs,
    split_anchor_id,
)
from project_paths import ALPASIM_DATA_ROOT


CASES = (
    ("test_clip_001_9306612661000", "front_tele"),
    ("test_clip_001_9306612661000", "front_wide"),
    ("test_clip_063_18787721418000", "front_tele"),
    ("test_clip_564_95233721166000", "cross_left"),
    ("test_clip_316_1404486225261000", "cross_right"),
)
CANDIDATE_RASTER = (480, 270)
REFERENCE_RASTER = (640, 360)
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
NEAR_PLANE_M = 1e-3
DEPTH_TOLERANCE_M = 1e-9
OUTPUT = Path("/tmp/step7e_multicase_occlusion_resolution_profile.json")


def run_resolution(
    *,
    actors,
    ego,
    calibration,
    camera_name,
    image_width,
    image_height,
    raster,
):
    raster_width, raster_height = raster
    return build_actor_camera_occlusion_from_geometry(
        actors=actors,
        recorded_ego_message=ego.message,
        calibration=calibration,
        camera_name=camera_name,
        image_width_px=image_width,
        image_height_px=image_height,
        raster_width=raster_width,
        raster_height=raster_height,
        maximum_depth=MAXIMUM_DEPTH,
        maximum_boundary_extent_px=MAXIMUM_BOUNDARY_EXTENT_PX,
        near_plane_m=NEAR_PLANE_M,
        depth_tolerance_m=DEPTH_TOLERANCE_M,
    )


def main() -> int:
    case_rows = []
    all_comparable_deltas = []
    aggregate_surface_change_ids = []
    aggregate_winning_change_ids = []
    aggregate_occluder_change_ids = []
    aggregate_stability_status_counts = Counter()
    camera_case_counts = Counter()

    for case_index, (anchor_id, camera_name) in enumerate(CASES, start=1):
        clip_id, anchor_ns = split_anchor_id(anchor_id)
        reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
        ego, actors = exact_inputs(reader, anchor_ns)

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

        candidate = run_resolution(
            actors=actors,
            ego=ego,
            calibration=calibration,
            camera_name=camera_name,
            image_width=frame.value.width,
            image_height=frame.value.height,
            raster=CANDIDATE_RASTER,
        )
        reference = run_resolution(
            actors=actors,
            ego=ego,
            calibration=calibration,
            camera_name=camera_name,
            image_width=frame.value.width,
            image_height=frame.value.height,
            raster=REFERENCE_RASTER,
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
                f"Unresolved boundaries for {anchor_id} {camera_name}: "
                f"candidate={candidate_unresolved}, "
                f"reference={reference_unresolved}"
            )

        profile = build_occlusion_resolution_profile(
            candidate_evidence=candidate.occlusion.actor_evidence,
            reference_evidence=reference.occlusion.actor_evidence,
        )
        summary = profile.summary
        if summary.actor_count != len(actors):
            raise RuntimeError(
                f"Profile Actor count mismatch: {anchor_id} {camera_name}"
            )
        if len(profile.stability_evidence) != len(profile.comparisons):
            raise RuntimeError(
                f"Stability evidence count mismatch: {anchor_id} {camera_name}"
            )
        comparison_ids = tuple(
            item.track_id for item in profile.comparisons
        )
        stability_ids = tuple(
            item.track_id for item in profile.stability_evidence
        )
        if stability_ids != comparison_ids:
            raise RuntimeError(
                f"Stability evidence Actor IDs differ: {anchor_id} {camera_name}"
            )

        case_stability_status_counts = Counter(
            item.stability_status
            for item in profile.stability_evidence
        )
        aggregate_stability_status_counts.update(
            case_stability_status_counts
        )

        comparable_deltas = [
            item.absolute_visible_fraction_delta
            for item in profile.comparisons
            if item.both_evaluated
        ]
        all_comparable_deltas.extend(comparable_deltas)
        aggregate_surface_change_ids.extend(
            f"{anchor_id}/{camera_name}/{track_id}"
            for track_id in summary.sampled_surface_presence_change_track_ids
        )
        aggregate_winning_change_ids.extend(
            f"{anchor_id}/{camera_name}/{track_id}"
            for track_id in summary.winning_presence_change_track_ids
        )
        aggregate_occluder_change_ids.extend(
            f"{anchor_id}/{camera_name}/{track_id}"
            for track_id in summary.occluding_actor_set_change_track_ids
        )
        camera_case_counts[camera_name] += 1

        case_row = {
            "anchor_id": anchor_id,
            "clip_id": clip_id,
            "camera_name": camera_name,
            "input_actor_count": len(actors),
            "candidate_unresolved_boundary_count": candidate_unresolved,
            "reference_unresolved_boundary_count": reference_unresolved,
            "summary": summary.to_dict(),
            "stability_status_counts": dict(
                sorted(case_stability_status_counts.items())
            ),
            "comparisons": [
                item.to_dict() for item in profile.comparisons
            ],
            "stability_evidence": [
                item.to_dict() for item in profile.stability_evidence
            ],
        }
        case_rows.append(case_row)

        print(f"Case {case_index}/{len(CASES)}: {anchor_id} {camera_name}")
        print(
            f"  actors={summary.actor_count} "
            f"comparable={summary.comparable_actor_count} "
            f"surface_changes={summary.sampled_surface_presence_change_count} "
            f"winner_changes={summary.winning_presence_change_count} "
            f"occluder_changes={summary.occluding_actor_set_change_count} "
            f"nonzero_deltas={summary.nonzero_visible_fraction_delta_count} "
            f"max_delta={summary.maximum_absolute_visible_fraction_delta} "
            f"median_delta={summary.median_absolute_visible_fraction_delta}"
        )
        print(
            "  stability_status_counts=",
            dict(sorted(case_stability_status_counts.items())),
        )
        if summary.winning_presence_change_track_ids:
            print(
                "  winning_presence_change_tracks=",
                list(summary.winning_presence_change_track_ids),
            )
        if summary.occluding_actor_set_change_track_ids:
            print(
                "  occluder_set_change_tracks=",
                list(summary.occluding_actor_set_change_track_ids),
            )

    aggregate = {
        "case_count": len(case_rows),
        "camera_case_counts": dict(sorted(camera_case_counts.items())),
        "actor_comparison_count": sum(
            row["summary"]["actor_count"] for row in case_rows
        ),
        "comparable_actor_count": sum(
            row["summary"]["comparable_actor_count"] for row in case_rows
        ),
        "sampled_surface_presence_change_count": len(
            aggregate_surface_change_ids
        ),
        "winning_presence_change_count": len(aggregate_winning_change_ids),
        "occluding_actor_set_change_count": len(aggregate_occluder_change_ids),
        "nonzero_visible_fraction_delta_count": sum(
            row["summary"]["nonzero_visible_fraction_delta_count"]
            for row in case_rows
        ),
        "stability_status_counts": dict(
            sorted(aggregate_stability_status_counts.items())
        ),
        "maximum_absolute_visible_fraction_delta": (
            max(all_comparable_deltas) if all_comparable_deltas else None
        ),
        "median_absolute_visible_fraction_delta": (
            median(all_comparable_deltas) if all_comparable_deltas else None
        ),
        "sampled_surface_presence_change_actor_keys": sorted(
            aggregate_surface_change_ids
        ),
        "winning_presence_change_actor_keys": sorted(
            aggregate_winning_change_ids
        ),
        "occluding_actor_set_change_actor_keys": sorted(
            aggregate_occluder_change_ids
        ),
    }

    report = {
        "schema_version": "step7e-multicase-occlusion-resolution-profile-v01",
        "candidate_raster": list(CANDIDATE_RASTER),
        "reference_raster": list(REFERENCE_RASTER),
        "maximum_depth": MAXIMUM_DEPTH,
        "maximum_boundary_extent_px": MAXIMUM_BOUNDARY_EXTENT_PX,
        "near_plane_m": NEAR_PLANE_M,
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
    print("  winning_presence_change_actor_keys:")
    for value in aggregate["winning_presence_change_actor_keys"]:
        print("   ", value)
    print("  occluding_actor_set_change_actor_keys:")
    for value in aggregate["occluding_actor_set_change_actor_keys"]:
        print("   ", value)
    print("Output:", OUTPUT)
    print(
        "PASS: multicase threshold-free occlusion resolution profile "
        "completed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
