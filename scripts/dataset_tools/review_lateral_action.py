#!/usr/bin/env python3
"""Generate a lightweight front-wide review video for one lateral-action case.

The tool is intentionally small and fast:
- reads existing JSONL feature/scan files;
- discovers front_wide image files for one clip;
- selects frames in a configurable time window;
- invokes the system ffmpeg executable directly;
- prints the map/trajectory evidence beside the output path.

It does not modify raw clip data and does not rerun lane matching.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path("/home/lab/data_from_alpasim")
DEFAULT_LATERAL_FEATURES = (
    ROOT / "annotations" / "v0.1-draft" / "intermediate"
    / "lateral_action_features_v0.2.jsonl"
)
DEFAULT_SCAN_CASES = ROOT / "reports" / "lateral_threshold_scan_cases_v0.1.jsonl"
DEFAULT_OUTPUT_DIRECTORY = ROOT / "reports" / "lateral_action_reviews"
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(\d{10,19})(?!\d)")


@dataclass(frozen=True, slots=True)
class TimestampedImage:
    stamp_ns: int
    path: Path


def read_jsonl_record(path: Path, anchor_id: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            if str(value.get("anchor_id", "")) == anchor_id:
                return value
    return None


def extract_timestamp_ns(path: Path) -> int | None:
    """Extract the last 10-19 digit integer from a filename stem."""
    matches = TIMESTAMP_PATTERN.findall(path.stem)
    if not matches:
        return None
    return int(matches[-1])


def _timestamp_ns_from_row(row: Mapping[str, Any]) -> int:
    for key in (
        "timestamp_ns",
        "stamp_ns",
        "time_ns",
    ):
        value = row.get(key)

        if isinstance(value, int) and not isinstance(value, bool):
            return value

    for key in (
        "stamp",
        "timestamp",
    ):
        value = row.get(key)

        if not isinstance(value, Mapping):
            continue

        sec = value.get("sec")
        nanosec = value.get("nanosec")

        if (
            isinstance(sec, int)
            and not isinstance(sec, bool)
            and isinstance(nanosec, int)
            and not isinstance(nanosec, bool)
        ):
            return (
                sec * 1_000_000_000
                + nanosec
            )

    raise ValueError(
        "camera timestamp row does not contain "
        "a supported timestamp"
    )


def discover_front_wide_images(
    clip_path: Path,
) -> tuple[TimestampedImage, ...]:
    """Read front_wide frames from its timestamp index."""

    camera_directory = (
        clip_path
        / "cameras"
        / "front_wide"
    )

    timestamp_path = (
        camera_directory
        / "timestamps.jsonl"
    )

    if not timestamp_path.is_file():
        raise FileNotFoundError(
            "front_wide timestamp index not found: "
            + str(timestamp_path)
        )

    images: list[TimestampedImage] = []

    with timestamp_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            row = json.loads(line)

            if not isinstance(row, dict):
                raise ValueError(
                    f"{timestamp_path}:{line_number} "
                    "is not a JSON object"
                )

            stamp_ns = _timestamp_ns_from_row(
                row
            )

            image_name = row.get(
                "image_path"
            )

            if not isinstance(
                image_name,
                str,
            ):
                raise ValueError(
                    f"{timestamp_path}:{line_number} "
                    "has no valid image_path"
                )

            image_path = Path(image_name)

            if not image_path.is_absolute():
                image_path = (
                    camera_directory
                    / image_path
                )

            if not image_path.is_file():
                raise FileNotFoundError(
                    "indexed front_wide image "
                    "does not exist: "
                    + str(image_path)
                )

            images.append(
                TimestampedImage(
                    stamp_ns=stamp_ns,
                    path=image_path,
                )
            )

    images.sort(
        key=lambda item: (
            item.stamp_ns,
            str(item.path),
        )
    )

    return tuple(images)


def select_window(
    images: Sequence[TimestampedImage],
    start_ns: int,
    end_ns: int,
) -> tuple[TimestampedImage, ...]:
    if end_ns <= start_ns:
        raise ValueError("review window end must be after start")
    return tuple(item for item in images if start_ns <= item.stamp_ns <= end_ns)


def ffconcat_escape(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def frame_duration_seconds(
    current_ns: int,
    next_ns: int | None,
    fallback_fps: float,
) -> float:
    if fallback_fps <= 0.0:
        raise ValueError("fallback_fps must be positive")
    if next_ns is None or next_ns <= current_ns:
        return 1.0 / fallback_fps
    return max(0.01, min(1.0, (next_ns - current_ns) / 1e9))


def write_concat_file(
    images: Sequence[TimestampedImage],
    path: Path,
    fallback_fps: float,
) -> None:
    if not images:
        raise ValueError("cannot write an empty review video")
    lines = ["ffconcat version 1.0"]
    for index, image in enumerate(images):
        next_ns = images[index + 1].stamp_ns if index + 1 < len(images) else None
        duration = frame_duration_seconds(image.stamp_ns, next_ns, fallback_fps)
        lines.append(f"file '{ffconcat_escape(image.path)}'")
        lines.append(f"duration {duration:.9f}")
    # ffmpeg concat requires the final file repeated to honor its duration.
    lines.append(f"file '{ffconcat_escape(images[-1].path)}'")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ffmpeg(
    concat_path: Path,
    output_path: Path,
    *,
    crf: int,
    preset: str,
    maximum_width: int,
) -> None:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("ffmpeg was not found in PATH")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scale_filter = (
        f"scale='min({maximum_width},iw)':-2:flags=fast_bilinear,"
        "format=yuv420p"
    )
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-vf",
        scale_filter,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def print_case_summary(
    anchor_id: str,
    feature: Mapping[str, Any],
    scan_case: Mapping[str, Any] | None,
    frame_count: int,
    output_path: Path,
) -> None:
    lateral = feature["lateral"]
    topology = lateral["topology"]
    ego_deg = math.degrees(float(lateral["trajectory_yaw_signed_change_rad"]))
    map_value = lateral.get("map_corridor_heading_change_rad")
    map_deg = math.degrees(float(map_value)) if map_value is not None else None
    signed_direction = "positive" if ego_deg > 0.0 else "negative"

    print("Anchor ID:", anchor_id)
    print("Clip:", feature["clip_id"])
    print("Anchor ns:", feature["anchor_ns"])
    print("Junction level:", topology["junction_evidence_level"])
    print("Junction reasons:", topology["junction_evidence_reasons"])
    print("Ego yaw change deg:", round(ego_deg, 3))
    print("Map heading change deg:", None if map_deg is None else round(map_deg, 3))
    print("Signed direction candidate:", signed_direction, "(left/right not yet verified)")
    print("Lateral quality gate:", lateral["lateral_quality_gate"])
    print("Contains adjacent transition:", lateral["contains_adjacent_transition"])
    print("Lane sequence:", lateral["lane_sequence"])
    if scan_case is not None:
        print("Scan source:", scan_case.get("source"))
        print("Scan ego threshold deg:", scan_case.get("ego_threshold_deg"))
        print("Scan map threshold deg:", scan_case.get("map_threshold_deg"))
    print("Video frames:", frame_count)
    print("Review video:", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a front-wide video for one lateral-action review case."
    )
    parser.add_argument("--anchor-id", required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--feature-input", type=Path, default=DEFAULT_LATERAL_FEATURES)
    parser.add_argument("--scan-case-input", type=Path, default=DEFAULT_SCAN_CASES)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--pre-sec", type=float, default=0.5)
    parser.add_argument("--post-sec", type=float, default=3.0)
    parser.add_argument("--fallback-fps", type=float, default=10.0)
    parser.add_argument("--maximum-width", type=int, default=1280)
    parser.add_argument("--crf", type=int, default=25)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pre_sec < 0.0 or args.post_sec <= 0.0:
        raise ValueError("--pre-sec must be non-negative and --post-sec positive")
    if args.maximum_width <= 0:
        raise ValueError("--maximum-width must be positive")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be in [0, 51]")

    feature = read_jsonl_record(args.feature_input.expanduser().resolve(), args.anchor_id)
    if feature is None:
        raise KeyError(f"anchor not found in lateral features: {args.anchor_id}")
    scan_case = read_jsonl_record(
        args.scan_case_input.expanduser().resolve(), args.anchor_id
    )

    clip_id = str(feature["clip_id"])
    anchor_ns = int(feature["anchor_ns"])
    clip_path = args.dataset_root.expanduser().resolve() / clip_id
    images = discover_front_wide_images(clip_path)
    if not images:
        raise RuntimeError(f"no timestamped front_wide images found under {clip_path}")

    start_ns = anchor_ns - round(args.pre_sec * 1e9)
    end_ns = anchor_ns + round(args.post_sec * 1e9)
    selected = select_window(images, start_ns, end_ns)
    if len(selected) < 2:
        raise RuntimeError(
            f"only {len(selected)} front_wide frames found in the requested window"
        )

    output_directory = args.output_directory.expanduser().resolve()
    output_path = output_directory / f"{args.anchor_id}_front_wide.mp4"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"review video exists: {output_path}; use --force")

    with tempfile.TemporaryDirectory(prefix="lateral_review_") as directory:
        concat_path = Path(directory) / "frames.ffconcat"
        write_concat_file(selected, concat_path, args.fallback_fps)
        run_ffmpeg(
            concat_path,
            output_path,
            crf=args.crf,
            preset=args.preset,
            maximum_width=args.maximum_width,
        )

    print_case_summary(args.anchor_id, feature, scan_case, len(selected), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
