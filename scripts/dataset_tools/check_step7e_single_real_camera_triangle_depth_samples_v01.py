#!/usr/bin/env python3
"""Validate one real in-FOV camera triangle through raster depth sampling.

Step 7E-2D diagnostic only. The script searches the known real boundary-case
anchors in deterministic order and selects the first prepared Actor triangle
whose three vertices project inside the FOV, whose projected triangle is
non-degenerate, and whose selected diagnostic raster yields at least one center
sample. It writes a JSON report under /tmp and does not modify annotations.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from actor_box_projection_v01 import actor_box_corners_in_rig
from camera_facing_box_surfaces_v01 import prepare_camera_facing_box_triangles
from camera_projection_v01 import load_camera_calibration
from camera_triangle_depth_samples_v01 import sample_camera_triangle_depths
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT

NEAR_PLANE_M = 1e-3
RASTER_WIDTH = 320
RASTER_HEIGHT = 180
OUTPUT = Path("/tmp/step7e_single_real_camera_triangle_depth_samples.json")

CANDIDATES = (
    ("test_clip_001_9306612661000", "front_tele", "13"),
    ("test_clip_001_9306612661000", "front_tele", "29"),
    ("test_clip_001_9306612661000", "front_wide", "27"),
    ("test_clip_063_18787721418000", "front_tele", "18"),
    ("test_clip_063_18787721418000", "front_tele", "98"),
    ("test_clip_185_222151025385000", "front_tele", "229"),
    ("test_clip_185_222151025385000", "front_tele", "238"),
    ("test_clip_410_2056042617247000", "front_tele", "157"),
)


def split_anchor_id(anchor_id: str) -> tuple[str, int]:
    clip_id, stamp_text = anchor_id.rsplit("_", 1)
    return clip_id, int(stamp_text)


def exact_inputs(reader: DrivingClipReader, stamp_ns: int):
    ego = reader.get_recorded_ego_state_at_or_before(stamp_ns)
    if ego is None or ego.stamp_ns != stamp_ns:
        raise RuntimeError(f"Exact recorded Ego state unavailable: {stamp_ns}")
    snapshots = reader.get_actor_snapshots(stamp_ns, duration_ns=0)
    if len(snapshots) != 1 or snapshots[0].stamp_ns != stamp_ns:
        raise RuntimeError(f"Exact Actor snapshot unavailable: {stamp_ns}")
    return ego, snapshots[0].message["actors"]


def find_actor(actors: list[dict[str, Any]], track_id: str) -> dict[str, Any]:
    matches = [actor for actor in actors if str(actor.get("track_id")) == track_id]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one Actor track={track_id}, found {len(matches)}"
        )
    return matches[0]


def projected_twice_area(projections) -> float:
    first, second, third = projections
    return abs(
        (second.u - first.u) * (third.v - first.v)
        - (second.v - first.v) * (third.u - first.u)
    )


def main() -> int:
    attempts: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None

    for anchor_id, camera_name, track_id in CANDIDATES:
        clip_id, stamp_ns = split_anchor_id(anchor_id)
        reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
        ego, actors = exact_inputs(reader, stamp_ns)
        actor = find_actor(actors, track_id)
        frame = reader.camera_indexes[camera_name].exact(stamp_ns)
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

        for triangle_index, item in enumerate(prepared):
            vertices = item.vertices_camera
            projections = tuple(
                calibration.project_camera_point(vertex)
                for vertex in vertices
            )
            reason = None
            if not all(projection.positive_z for projection in projections):
                reason = "nonpositive_projection"
            elif not all(projection.within_fov for projection in projections):
                reason = "vertex_outside_fov"
            elif not all(
                math.isfinite(projection.u) and math.isfinite(projection.v)
                for projection in projections
            ):
                reason = "nonfinite_projection"
            elif projected_twice_area(projections) <= 1e-9:
                reason = "degenerate_projection"

            attempt = {
                "anchor_id": anchor_id,
                "camera_name": camera_name,
                "track_id": track_id,
                "triangle_index": triangle_index,
                "reason": reason,
            }
            attempts.append(attempt)
            if reason is not None:
                continue

            samples = sample_camera_triangle_depths(
                vertices,
                calibration,
                image_width_px=frame.value.width,
                image_height_px=frame.value.height,
                raster_width=RASTER_WIDTH,
                raster_height=RASTER_HEIGHT,
                near_plane_m=NEAR_PLANE_M,
            )
            if not samples.center_sampled_depths:
                attempt["reason"] = "no_center_samples"
                continue

            depths = [sample.depth_m for sample in samples.center_sampled_depths]
            sample_u = [sample.sample_u_px for sample in samples.center_sampled_depths]
            sample_v = [sample.sample_v_px for sample in samples.center_sampled_depths]
            vertex_depths = [vertex.z for vertex in vertices]

            if not all(math.isfinite(depth) and depth > 0.0 for depth in depths):
                raise RuntimeError("Sample depths must be positive and finite")
            if not all(0.0 <= value <= frame.value.width for value in sample_u):
                raise RuntimeError("Sample u coordinates must be inside image bounds")
            if not all(0.0 <= value <= frame.value.height for value in sample_v):
                raise RuntimeError("Sample v coordinates must be inside image bounds")
            if min(depths) < min(vertex_depths) - 1e-9:
                raise RuntimeError("Sample depth is below minimum vertex depth")
            if max(depths) > max(vertex_depths) + 1e-9:
                raise RuntimeError("Sample depth exceeds maximum vertex depth")

            selected = {
                "anchor_id": anchor_id,
                "clip_id": clip_id,
                "stamp_ns": stamp_ns,
                "camera_name": camera_name,
                "track_id": track_id,
                "actor_class": str(actor.get("label_class", "")),
                "triangle_index": triangle_index,
                "image_width_px": frame.value.width,
                "image_height_px": frame.value.height,
                "raster_width": RASTER_WIDTH,
                "raster_height": RASTER_HEIGHT,
                "vertices_camera": [
                    {"x": vertex.x, "y": vertex.y, "z": vertex.z}
                    for vertex in vertices
                ],
                "vertex_projections": [
                    {"u": projection.u, "v": projection.v}
                    for projection in projections
                ],
                "projected_area_px2": 0.5 * projected_twice_area(projections),
                "conservative_cell_count": samples.conservative_cell_count,
                "center_sampled_cell_count": samples.center_sampled_cell_count,
                "minimum_vertex_depth_m": min(vertex_depths),
                "maximum_vertex_depth_m": max(vertex_depths),
                "minimum_sample_depth_m": min(depths),
                "maximum_sample_depth_m": max(depths),
                "minimum_sample_u_px": min(sample_u),
                "maximum_sample_u_px": max(sample_u),
                "minimum_sample_v_px": min(sample_v),
                "maximum_sample_v_px": max(sample_v),
                "first_five_samples": [
                    {
                        "column": sample.cell.column,
                        "row": sample.cell.row,
                        "sample_u_px": sample.sample_u_px,
                        "sample_v_px": sample.sample_v_px,
                        "depth_m": sample.depth_m,
                        "barycentric_weights": list(sample.barycentric_weights),
                    }
                    for sample in samples.center_sampled_depths[:5]
                ],
            }
            break

        if selected is not None:
            break

    if selected is None:
        raise RuntimeError("No suitable real in-FOV triangle produced depth samples")

    report = {
        "near_plane_m": NEAR_PLANE_M,
        "selection_policy": "first deterministic suitable triangle",
        "attempt_count": len(attempts),
        "attempts": attempts,
        "selected": selected,
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("Selected real triangle")
    print("  anchor:", selected["anchor_id"])
    print("  camera:", selected["camera_name"])
    print("  track:", selected["track_id"])
    print("  triangle_index:", selected["triangle_index"])
    print("  projected_area_px2:", selected["projected_area_px2"])
    print("  conservative_cells:", selected["conservative_cell_count"])
    print("  center_samples:", selected["center_sampled_cell_count"])
    print(
        "  vertex_depth_range_m:",
        selected["minimum_vertex_depth_m"],
        selected["maximum_vertex_depth_m"],
    )
    print(
        "  sample_depth_range_m:",
        selected["minimum_sample_depth_m"],
        selected["maximum_sample_depth_m"],
    )
    print("Output:", OUTPUT)
    print("PASS: one real in-FOV camera triangle produced valid raster depth samples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
