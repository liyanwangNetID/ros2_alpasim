#!/usr/bin/env python3
"""Export Step 7E v02 evidence with missing-surface projection context.

The exporter reads all selected Keyframes, runs the existing enhanced geometric
and Actor-to-Actor occlusion pipeline once per Anchor, prepares one v02 row per
Actor, and atomically writes compact JSONL plus summary JSON. The v01 product is
not modified, static-scene occlusion remains unevaluated, and no final visibility
threshold or label is introduced.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from actor_geometric_occlusion_export_inputs_v02 import (
    prepare_actor_geometric_occlusion_projection_context_export_inputs,
)
from actor_geometric_occlusion_export_writer_v02 import (
    write_actor_geometric_occlusion_projection_context_evidence,
)
from actor_geometric_occlusion_with_projection_context_v01 import (
    build_actor_geometric_occlusion_with_projection_context,
)
from export_step7e_geometric_occlusion_evidence_v01 import (
    KEYFRAME_PATH,
    exact_inputs,
    load_camera_inputs,
    read_keyframes,
)
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT


OUTPUT_PATH = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.jsonl"
SUMMARY_PATH = (
    ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.summary.json"
)
RASTER_WIDTH = 480
RASTER_HEIGHT = 270
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
NEAR_PLANE_M = 1e-3
MAXIMUM_CHORD_ERROR_PX = 1.0
MAXIMUM_ADAPTIVE_DEPTH = 14
DEPTH_TOLERANCE_M = 1e-9


def build_anchor_export_inputs(*, keyframe, reader):
    """Build deterministic v02 writer inputs for one exact Anchor."""

    anchor_id = str(keyframe["anchor_id"])
    anchor_ns = int(keyframe["anchor_ns"])
    ego, actors = exact_inputs(reader, anchor_ns)
    cameras = load_camera_inputs(reader, anchor_id, anchor_ns)
    result = build_actor_geometric_occlusion_with_projection_context(
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
    return prepare_actor_geometric_occlusion_projection_context_export_inputs(
        keyframe=keyframe,
        actors=actors,
        result=result,
    )


def export(*, keyframe_path: Path, output_path: Path, summary_path: Path):
    """Run v02 export for all Keyframes and return the saved summary."""

    started = time.monotonic()
    keyframes = read_keyframes(keyframe_path)
    prepared = []
    current_clip_id = None
    reader = None

    for index, keyframe in enumerate(keyframes, start=1):
        clip_id = str(keyframe["clip_id"])
        if clip_id != current_clip_id:
            current_clip_id = clip_id
            reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
        assert reader is not None
        prepared.extend(
            build_anchor_export_inputs(keyframe=keyframe, reader=reader)
        )
        if index % 10 == 0 or index == len(keyframes):
            print(
                f"Progress: {index}/{len(keyframes)} "
                f"rows={len(prepared)} "
                f"elapsed_sec={time.monotonic() - started:.1f}",
                flush=True,
            )

    summary = (
        write_actor_geometric_occlusion_projection_context_evidence(
            inputs=prepared,
            output_path=output_path,
            summary_path=summary_path,
        )
    )
    print("Keyframes:", len(keyframes))
    print("Evidence rows:", summary["row_count"])
    print("Evidence status counts:", summary["evidence_status_counts"])
    print("Rows by Actor class:", summary["rows_by_actor_class"])
    print(
        "Missing-surface projection contexts:",
        summary["missing_surface_projection_context_count"],
    )
    print(
        "Projection-context status counts:",
        summary["missing_surface_projection_context_status_counts"],
    )
    print("Output:", output_path)
    print("Summary:", summary_path)
    print(
        "PASS: Step 7E v02 evidence exported atomically with projection "
        "context and without final visibility thresholds."
    )
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyframes", type=Path, default=KEYFRAME_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    export(
        keyframe_path=arguments.keyframes.expanduser().resolve(),
        output_path=arguments.output.expanduser().resolve(),
        summary_path=arguments.summary.expanduser().resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
