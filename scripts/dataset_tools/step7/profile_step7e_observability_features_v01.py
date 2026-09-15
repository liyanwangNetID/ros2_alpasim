#!/usr/bin/env python3
"""Step 7E.1 profile continuous observability evidence.

This read-only diagnostic traverses all selected Keyframes, calls the Step 7D
production projection path, and summarizes valid geometric projections by:

- camera;
- Actor class;
- camera-depth band;
- image-boundary truncation state.

It intentionally does not classify Actors as observable and does not apply
minimum area, height, ratio, or maximum distance thresholds.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from step7.actor_box_image_projection_v01 import project_actor_box_to_camera
from step7.camera_projection_v01 import load_camera_calibration
from step2.clip_reader import DrivingClipReader
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT
from step7.scene_fact_schema_v01 import CAMERA_NAMES

KEYFRAME_PATH = ANNOTATION_ROOT / "keyframes.jsonl"
OUTPUT_PATH = ANNOTATION_ROOT / "step7e_observability_profile_v01.json"

# Diagnostic reporting bands only. These do not define observability.
DEPTH_BAND_EDGES_M = (0.0, 10.0, 20.0, 40.0, 80.0, 160.0, math.inf)
PERCENTILES = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0)


@dataclass
class Distribution:
    values: list[float] = field(default_factory=list)

    def add(self, value: float) -> None:
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("distribution value must be finite")
        self.values.append(converted)

    def summary(self) -> dict[str, Any]:
        if not self.values:
            return {"count": 0, "percentiles": {}}
        ordered = sorted(self.values)
        result: dict[str, float] = {}
        for fraction in PERCENTILES:
            index = round(fraction * (len(ordered) - 1))
            result[f"p{int(round(fraction * 100)):02d}"] = ordered[index]
        return {
            "count": len(ordered),
            "mean": sum(ordered) / len(ordered),
            "percentiles": result,
        }


@dataclass
class GroupStats:
    count: int = 0
    inside_image_hull_area_px: Distribution = field(default_factory=Distribution)
    projected_hull_area_px: Distribution = field(default_factory=Distribution)
    inside_image_hull_ratio: Distribution = field(default_factory=Distribution)
    projected_height_px: Distribution = field(default_factory=Distribution)
    minimum_depth_m: Distribution = field(default_factory=Distribution)
    camera_sample_count: Distribution = field(default_factory=Distribution)

    def add(self, projection) -> None:
        self.count += 1
        self.inside_image_hull_area_px.add(projection.inside_image_hull_area_px)
        self.projected_hull_area_px.add(projection.projected_hull_area_px)
        self.inside_image_hull_ratio.add(projection.inside_image_hull_ratio)
        self.projected_height_px.add(projection.projected_height_px)
        assert projection.minimum_depth_m is not None
        self.minimum_depth_m.add(projection.minimum_depth_m)
        self.camera_sample_count.add(projection.camera_sample_count)

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "inside_image_hull_area_px": self.inside_image_hull_area_px.summary(),
            "projected_hull_area_px": self.projected_hull_area_px.summary(),
            "inside_image_hull_ratio": self.inside_image_hull_ratio.summary(),
            "projected_height_px": self.projected_height_px.summary(),
            "minimum_depth_m": self.minimum_depth_m.summary(),
            "camera_sample_count": self.camera_sample_count.summary(),
        }


def read_keyframes() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with KEYFRAME_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def depth_band(value: float) -> str:
    for lower, upper in zip(DEPTH_BAND_EDGES_M, DEPTH_BAND_EDGES_M[1:]):
        if lower <= value < upper:
            upper_label = "inf" if math.isinf(upper) else f"{upper:g}"
            return f"[{lower:g},{upper_label})"
    raise ValueError(f"Unexpected depth: {value}")


def truncation_band(ratio: float) -> str:
    if ratio >= 1.0 - 1e-9:
        return "not_truncated"
    if ratio >= 0.75:
        return "truncated_75_to_100_percent_inside"
    if ratio >= 0.50:
        return "truncated_50_to_75_percent_inside"
    if ratio >= 0.25:
        return "truncated_25_to_50_percent_inside"
    return "truncated_below_25_percent_inside"


def add_group(groups: dict[str, GroupStats], key: str, projection) -> None:
    groups.setdefault(key, GroupStats()).add(projection)


def summarize_groups(groups: dict[str, GroupStats]) -> dict[str, Any]:
    return {
        key: groups[key].summary()
        for key in sorted(groups)
    }


def main() -> int:
    started = time.monotonic()
    keyframes = read_keyframes()
    current_clip_id: str | None = None
    reader: DrivingClipReader | None = None
    calibrations = {}

    overall = GroupStats()
    by_camera: dict[str, GroupStats] = {}
    by_class: dict[str, GroupStats] = {}
    by_camera_class: dict[str, GroupStats] = {}
    by_depth: dict[str, GroupStats] = {}
    by_camera_depth: dict[str, GroupStats] = {}
    by_truncation: dict[str, GroupStats] = {}
    by_camera_truncation: dict[str, GroupStats] = {}

    actor_count = 0
    pair_count = 0
    failure_reasons: Counter[str] = Counter()
    processing_failures: list[dict[str, str]] = []

    for index, keyframe in enumerate(keyframes, start=1):
        clip_id = str(keyframe["clip_id"])
        anchor_id = str(keyframe["anchor_id"])
        anchor_ns = int(keyframe["anchor_ns"])

        try:
            if clip_id != current_clip_id:
                current_clip_id = clip_id
                reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
                calibrations = {}
                for camera_name in CAMERA_NAMES:
                    exact = reader.camera_indexes[camera_name].exact(anchor_ns)
                    if exact is None:
                        raise RuntimeError(f"No exact frame: {camera_name}")
                    frame = exact.value
                    calibrations[camera_name] = load_camera_calibration(
                        reader.clip_directory / "calibration" / f"{camera_name}.json",
                        camera_name=camera_name,
                        source_width=frame.width,
                        source_height=frame.height,
                    )

            assert reader is not None
            ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
            snapshots = reader.get_actor_snapshots(anchor_ns, duration_ns=0)
            if ego is None or ego.stamp_ns != anchor_ns:
                raise RuntimeError("Exact Ego state unavailable")
            if len(snapshots) != 1 or snapshots[0].stamp_ns != anchor_ns:
                raise RuntimeError("Exact Actor snapshot unavailable")
            actors = snapshots[0].message.get("actors")
            if not isinstance(actors, list):
                raise RuntimeError("Actor list unavailable")
            actor_count += len(actors)

            for camera_name in CAMERA_NAMES:
                calibration = calibrations[camera_name]
                for actor in actors:
                    pair_count += 1
                    projection = project_actor_box_to_camera(
                        actor,
                        recorded_ego_message=ego.message,
                        calibration=calibration,
                    )
                    if not projection.projection_valid:
                        failure_reasons[
                            projection.failure_reason or "missing_failure_reason"
                        ] += 1
                        continue

                    actor_class = str(actor.get("label_class"))
                    assert projection.minimum_depth_m is not None
                    distance_group = depth_band(projection.minimum_depth_m)
                    truncation_group = truncation_band(
                        projection.inside_image_hull_ratio
                    )

                    overall.add(projection)
                    add_group(by_camera, camera_name, projection)
                    add_group(by_class, actor_class, projection)
                    add_group(
                        by_camera_class,
                        f"{camera_name}|{actor_class}",
                        projection,
                    )
                    add_group(by_depth, distance_group, projection)
                    add_group(
                        by_camera_depth,
                        f"{camera_name}|{distance_group}",
                        projection,
                    )
                    add_group(by_truncation, truncation_group, projection)
                    add_group(
                        by_camera_truncation,
                        f"{camera_name}|{truncation_group}",
                        projection,
                    )

        except Exception as exc:
            processing_failures.append(
                {
                    "anchor_id": anchor_id,
                    "clip_id": clip_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

        if index % 100 == 0 or index == len(keyframes):
            print(
                f"Progress: {index}/{len(keyframes)} "
                f"pairs={pair_count} valid={overall.count} "
                f"elapsed_sec={time.monotonic() - started:.1f}",
                flush=True,
            )

    report = {
        "schema_version": "step7e-observability-profile-v01",
        "description": (
            "Continuous geometric projection evidence only; no observability "
            "classification and no Actor-to-Actor occlusion handling."
        ),
        "source": {
            "keyframes_path": str(KEYFRAME_PATH),
            "keyframe_count": len(keyframes),
            "actor_count": actor_count,
            "actor_camera_pair_count": pair_count,
            "valid_projection_count": overall.count,
            "invalid_projection_count": sum(failure_reasons.values()),
            "invalid_projection_reasons": dict(sorted(failure_reasons.items())),
        },
        "projection_policy": {
            "sampling": "adaptive",
            "maximum_chord_error_px": 1.0,
            "maximum_adaptive_depth": 14,
            "near_plane_m": 1e-3,
        },
        "reporting_depth_band_edges_m": [
            None if math.isinf(value) else value
            for value in DEPTH_BAND_EDGES_M
        ],
        "percentiles": list(PERCENTILES),
        "overall": overall.summary(),
        "by_camera": summarize_groups(by_camera),
        "by_actor_class": summarize_groups(by_class),
        "by_camera_and_actor_class": summarize_groups(by_camera_class),
        "by_minimum_depth_band_m": summarize_groups(by_depth),
        "by_camera_and_minimum_depth_band_m": summarize_groups(by_camera_depth),
        "by_image_truncation_band": summarize_groups(by_truncation),
        "by_camera_and_image_truncation_band": summarize_groups(
            by_camera_truncation
        ),
        "processing_failures": processing_failures,
        "elapsed_seconds": time.monotonic() - started,
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("Keyframes:", len(keyframes))
    print("Actors:", actor_count)
    print("Actor/camera pairs:", pair_count)
    print("Valid projections:", overall.count)
    print("Invalid projections:", sum(failure_reasons.values()))
    print("Invalid reasons:", dict(sorted(failure_reasons.items())))
    print("Processing failures:", len(processing_failures))
    print("Output:", OUTPUT_PATH)

    assert pair_count == actor_count * len(CAMERA_NAMES)
    assert overall.count + sum(failure_reasons.values()) == pair_count
    assert not processing_failures
    assert "missing_failure_reason" not in failure_reasons

    print("PASS: Step 7E.1 continuous observability profile generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
