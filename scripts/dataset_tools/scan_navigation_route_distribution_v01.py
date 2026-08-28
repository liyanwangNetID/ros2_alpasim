#!/usr/bin/env python3
"""Scan Step 6 route-feature distributions before freezing navigation rules."""
from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/home/lab/data_from_alpasim/annotations/v0.1-draft")
FEATURE_PATH = ROOT / "intermediate/navigation_route_features_v0.1.jsonl"
KEYFRAME_PATH = ROOT / "keyframes.jsonl"


def read_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            anchor_id = str(record["anchor_id"])
            if anchor_id in result:
                raise ValueError(
                    f"duplicate Anchor at {path}:{line_number}: {anchor_id}"
                )
            result[anchor_id] = record
    return result


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def describe(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "p01": percentile(values, 0.01),
        "p05": percentile(values, 0.05),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "median": statistics.median(values) if values else None,
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values) if values else None,
    }


def print_description(name: str, values: list[float], unit: str = "") -> None:
    print()
    print(name)
    for key, value in describe(values).items():
        suffix = unit if value is not None and key != "count" else ""
        print(f"  {key}: {value}{suffix}")


def main() -> int:
    features = read_index(FEATURE_PATH)
    keyframes = read_index(KEYFRAME_PATH)
    if set(features) != set(keyframes):
        raise ValueError("Navigation feature and Keyframe Anchor sets differ")

    quality_counts = Counter()
    unknown_reasons = Counter()
    frame_counts = Counter()
    point_counts = Counter()
    intersection_counts = Counter()
    geometry_values: dict[str, list[float]] = defaultdict(list)
    signed_heading_by_lateral: dict[str, list[float]] = defaultdict(list)
    final_y_by_lateral: dict[str, list[float]] = defaultdict(list)
    event_context_counts = Counter()

    for anchor_id in sorted(features):
        feature = features[anchor_id]
        keyframe = keyframes[anchor_id]
        status = str(feature["quality_status"])
        quality_counts[status] += 1
        frame_counts[str(feature.get("route_frame_id"))] += 1

        for reason in feature.get("reasons", []):
            unknown_reasons[str(reason)] += 1

        event_types = {str(event["type"]) for event in keyframe.get("events", [])}
        has_intersection_context = bool(
            event_types & {"junction_approach", "junction_entry", "turn_start"}
        )
        intersection_counts[
            "intersection_context" if has_intersection_context else "no_intersection_context"
        ] += 1

        if status != "usable":
            continue

        route = feature["route"]
        point_counts[int(route["valid_point_count"])] += 1
        lateral_action = str(keyframe["meta_action"]["lateral"])

        signed_deg = math.degrees(float(route["route_signed_heading_change_rad"]))
        absolute_deg = math.degrees(float(route["route_absolute_heading_change_rad"]))
        excursion_deg = math.degrees(float(route["route_maximum_heading_excursion_rad"]))
        final_y = float(route["final_local_y_m"])

        geometry_values["signed_heading_change_deg"].append(signed_deg)
        geometry_values["absolute_heading_change_deg"].append(absolute_deg)
        geometry_values["maximum_heading_excursion_deg"].append(excursion_deg)
        geometry_values["final_local_y_m"].append(final_y)
        geometry_values["route_path_length_m"].append(float(route["route_path_length_m"]))
        geometry_values["forward_point_fraction"].append(float(route["forward_point_fraction"]))
        geometry_values["route_age_ms"].append(float(feature["route_age_ns"]) / 1e6)

        signed_heading_by_lateral[lateral_action].append(signed_deg)
        final_y_by_lateral[lateral_action].append(final_y)

        coarse_direction = (
            "left" if signed_deg > 0.0 else "right" if signed_deg < 0.0 else "zero"
        )
        event_context_counts[(
            "intersection" if has_intersection_context else "non_intersection",
            coarse_direction,
        )] += 1

    print("=" * 78)
    print("STEP 6 ROUTE FEATURE DISTRIBUTION")
    print("=" * 78)
    print("Records:", len(features))
    print("Quality:", dict(quality_counts))
    print("Unknown reasons:", dict(unknown_reasons))
    print("Route frames:", dict(frame_counts))
    print("Valid-point counts:", dict(sorted(point_counts.items())))
    print("Context counts:", dict(intersection_counts))

    for name in (
        "route_age_ms",
        "route_path_length_m",
        "forward_point_fraction",
        "signed_heading_change_deg",
        "absolute_heading_change_deg",
        "maximum_heading_excursion_deg",
        "final_local_y_m",
    ):
        unit = " deg" if name.endswith("_deg") else ""
        print_description(name, geometry_values[name], unit)

    print()
    print("=" * 78)
    print("DIAGNOSTIC CROSS-TAB WITH META-ACTION")
    print("Not used for Navigation generation")
    print("=" * 78)
    for action in sorted(signed_heading_by_lateral):
        print()
        print("Lateral action:", action)
        print("  Anchor count:", len(signed_heading_by_lateral[action]))
        print("  Signed heading deg:", describe(signed_heading_by_lateral[action]))
        print("  Final local y m:", describe(final_y_by_lateral[action]))

    print()
    print("Direction signs by context:")
    for key, count in sorted(event_context_counts.items()):
        print(" ", key[0], "+", key[1], ":", count)

    print()
    print("PASS: Step 6 route distribution scan completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
