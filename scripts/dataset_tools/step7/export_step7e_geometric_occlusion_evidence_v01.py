#!/usr/bin/env python3
"""Export complete Step 7E geometric-occlusion evidence for all Keyframes.

The exporter reads selected Keyframes, loads exact current Ego, Actor, and
four-camera inputs, runs the complete threshold-free geometric/Actor-occlusion
pipeline once per Anchor, then atomically writes one compact JSONL row per
Anchor/Actor plus a summary JSON. Static-scene occlusion and final visibility
labels remain intentionally unevaluated.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from step7.actor_geometric_occlusion_export_inputs_v01 import (
    prepare_actor_geometric_occlusion_export_inputs,
)
from step7.actor_geometric_occlusion_export_writer_v01 import (
    ActorGeometricOcclusionExportInput,
    write_actor_geometric_occlusion_evidence,
)
from step7.actor_geometric_occlusion_from_geometry_v01 import (
    CameraGeometricOcclusionBuildInput,
    build_actor_geometric_occlusion_from_geometry,
)
from step7.camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT
from step7.scene_fact_schema_v01 import CAMERA_NAMES


KEYFRAME_PATH = ANNOTATION_ROOT / "keyframes.jsonl"
OUTPUT_PATH = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v01.jsonl"
SUMMARY_PATH = (
    ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v01.summary.json"
)
RASTER_WIDTH = 480
RASTER_HEIGHT = 270
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
NEAR_PLANE_M = 1e-3
MAXIMUM_CHORD_ERROR_PX = 1.0
MAXIMUM_ADAPTIVE_DEPTH = 14
DEPTH_TOLERANCE_M = 1e-9


def read_keyframes(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("anchor_id", "clip_id", "anchor_ns"):
                if field not in record:
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            records.append(record)
    if not records:
        raise RuntimeError(f"No Keyframes found in {path}")
    identities = tuple(str(item["anchor_id"]) for item in records)
    if len(set(identities)) != len(identities):
        raise ValueError("Keyframe anchor_id values must be unique")
    return sorted(records, key=lambda item: str(item["anchor_id"]))


def exact_inputs(reader: DrivingClipReader, anchor_ns: int):
    ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
    if ego is None or ego.stamp_ns != anchor_ns:
        raise RuntimeError("Exact recorded Ego state unavailable")
    snapshots = reader.get_actor_snapshots(anchor_ns, duration_ns=0)
    if len(snapshots) != 1 or snapshots[0].stamp_ns != anchor_ns:
        raise RuntimeError("Exact Actor snapshot unavailable")
    actors = snapshots[0].message.get("actors")
    if not isinstance(actors, list):
        raise RuntimeError("Actor snapshot has no actors list")
    return ego, actors


def load_camera_inputs(
    reader: DrivingClipReader,
    anchor_id: str,
    anchor_ns: int,
) -> dict[str, CameraGeometricOcclusionBuildInput]:
    cameras = {}
    for camera_name in CAMERA_NAMES:
        exact = reader.camera_indexes[camera_name].exact(anchor_ns)
        if exact is None:
            raise RuntimeError(
                f"No exact camera frame for {anchor_id} {camera_name}"
            )
        frame = exact.value
        calibration = load_camera_calibration(
            reader.clip_directory / "calibration" / f"{camera_name}.json",
            camera_name=camera_name,
            source_width=frame.width,
            source_height=frame.height,
        )
        if calibration.max_angle_rad is None:
            raise RuntimeError(
                f"Camera calibration has no max_angle_rad: {camera_name}"
            )
        cameras[camera_name] = CameraGeometricOcclusionBuildInput(
            calibration=calibration,
            image_width_px=frame.width,
            image_height_px=frame.height,
        )
    return cameras


def build_anchor_export_inputs(
    *,
    keyframe: dict[str, Any],
    reader: DrivingClipReader,
) -> tuple[ActorGeometricOcclusionExportInput, ...]:
    anchor_id = str(keyframe["anchor_id"])
    anchor_ns = int(keyframe["anchor_ns"])
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
    return prepare_actor_geometric_occlusion_export_inputs(
        keyframe=keyframe,
        actors=actors,
        pipeline_result=result.combined,
    )


def export(
    *,
    keyframe_path: Path,
    output_path: Path,
    summary_path: Path,
) -> None:
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

    summary = write_actor_geometric_occlusion_evidence(
        inputs=prepared,
        output_path=output_path,
        summary_path=summary_path,
    )
    print("Keyframes:", len(keyframes))
    print("Evidence rows:", summary.row_count)
    print("Evidence status counts:", dict(summary.evidence_status_counts))
    print("Rows by Actor class:", dict(summary.rows_by_actor_class))
    print("Output:", output_path)
    print("Summary:", summary_path)
    print(
        "PASS: Step 7E geometric-occlusion evidence exported atomically "
        "without final visibility thresholds."
    )


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
