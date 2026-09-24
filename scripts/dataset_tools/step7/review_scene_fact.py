#!/usr/bin/env python3
"""Create a four-camera video and print one final Step 7 Scene-Fact."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT, REPORT_ROOT
from step2.clip_reader import CAMERA_NAMES, DrivingClipReader

DEFAULT_SCENE_FACTS = ANNOTATION_ROOT / "scene_facts.jsonl"
DEFAULT_OUTPUT_DIRECTORY = REPORT_ROOT / "step7_scene_fact_reviews"
CAMERA_TOLERANCE_NS = 60_000_000
CAMERA_LABELS = {
    "front_wide": "FRONT WIDE",
    "front_tele": "FRONT TELE",
    "cross_left": "CROSS LEFT",
    "cross_right": "CROSS RIGHT",
}


def read_scene_facts(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    anchors: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            anchor_id = str(row.get("anchor_id", ""))
            if not anchor_id:
                raise ValueError(f"{path}:{line_number} has no anchor_id")
            if anchor_id in anchors:
                raise ValueError(f"duplicate Scene-Fact Anchor: {anchor_id}")
            anchors.add(anchor_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"no Scene-Fact rows found: {path}")
    return tuple(rows)


def select_scene_fact(
    rows: Sequence[Mapping[str, Any]],
    *,
    anchor_id: str | None,
    seed: int | None,
    require_present_role: bool,
) -> Mapping[str, Any]:
    if anchor_id is not None:
        for row in rows:
            if str(row["anchor_id"]) == anchor_id:
                return row
        raise KeyError(f"Scene-Fact Anchor not found: {anchor_id}")
    candidates = list(rows)
    if require_present_role:
        roles = ("lead_actors", "left_nearby_actors", "right_nearby_actors")
        candidates = [
            row
            for row in candidates
            if (
                any(
                    row.get("actor_context", {}).get(role, ())
                    for role in roles
                )
            )
        ]
        if not candidates:
            raise ValueError("no Scene-Fact has a present selected Actor role")
    return random.Random(seed).choice(candidates)


def print_scene_fact(row: Mapping[str, Any]) -> None:
    print("=" * 72)
    print("STEP 7 FINAL SCENE-FACT HUMAN REVIEW")
    print("=" * 72)
    print("Anchor ID:", row["anchor_id"])
    print("Clip ID:", row["clip_id"])
    print("Anchor ns:", row["anchor_ns"])
    print("Format version:", row["scene_fact_format_version"])
    print("Rule version:", row["rule_version"])
    print()
    print(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=False))
    print()


def target_timestamps(
    reader: DrivingClipReader,
    *,
    start_ns: int,
    end_ns: int,
) -> tuple[int, ...]:
    index = reader.camera_indexes["front_wide"]
    matches = index.closed_range(start_ns, end_ns)
    stamps = tuple(match.timestamp_ns for match in matches)
    if len(stamps) < 2:
        raise RuntimeError("fewer than two front-wide frames exist in the review window")
    return stamps


def synchronized_paths(
    reader: DrivingClipReader,
    stamp_ns: int,
    *,
    tolerance_ns: int = CAMERA_TOLERANCE_NS,
) -> dict[str, Path] | None:
    """Return a complete four-camera set, or None for an incomplete timestamp."""
    paths: dict[str, Path] = {}
    for camera_name in CAMERA_NAMES:
        match = reader.camera_indexes[camera_name].nearest(
            stamp_ns,
            tolerance_ns=tolerance_ns,
        )
        if match is None:
            return None
        paths[camera_name] = match.value.image_path
    return paths


def letterbox(image, width: int, height: int):
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    canvas = cv2.copyMakeBorder(
        resized,
        0,
        height - resized_height,
        0,
        width - resized_width,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print one final Scene-Fact and create a four-camera review video."
    )
    parser.add_argument("--anchor-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--allow-empty-roles",
        action="store_true",
        help="Allow random selection of Scene-Facts with no selected Actor role.",
    )
    parser.add_argument("--scene-facts", type=Path, default=DEFAULT_SCENE_FACTS)
    parser.add_argument("--dataset-root", type=Path, default=ALPASIM_DATA_ROOT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--pre-sec", type=float, default=2.0)
    parser.add_argument("--post-sec", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.pre_sec < 0.0 or args.post_sec <= 0.0:
        raise ValueError("--pre-sec must be non-negative and --post-sec positive")
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive")
    if args.panel_width <= 0 or args.panel_height <= 0:
        raise ValueError("panel dimensions must be positive")
    rows = read_scene_facts(args.scene_facts.expanduser().resolve())
    row = select_scene_fact(
        rows,
        anchor_id=args.anchor_id,
        seed=args.seed,
        require_present_role=not args.allow_empty_roles,
    )
    print_scene_fact(row)
    if args.no_video:
        print("Video generation: skipped")
        return 0
    output_path = (
        args.output_directory.expanduser().resolve()
        / f"{row['anchor_id']}_scene_fact_review.mp4"
    )
    frame_count = create_review_video(
        row,
        dataset_root=args.dataset_root.expanduser().resolve(),
        output_path=output_path,
        pre_seconds=args.pre_sec,
        post_seconds=args.post_sec,
        fps=args.fps,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        force=args.force,
    )
    print("Video frames:", frame_count)
    print("Review video:", output_path)
    print("PASS: Scene-Fact review case generated.")
    return 0


# Frozen Actor-list review overlays
from dataclasses import replace
from step7.projection import load_camera_calibration, project_actor_box_to_camera

ROLE_LABEL_PREFIXES = {
    "lead_actors": "LEAD",
    "left_nearby_actors": "LEFT",
    "right_nearby_actors": "RIGHT",
}
REVIEW_BOX_COLORS = {
    "LEAD": (0, 0, 255),
    "LEFT": (0, 255, 255),
    "RIGHT": (0, 255, 0),
}

# Review-image alignment only. The recorded front streams have a vertical
# image-to-calibration offset not present in the cross-camera streams.
# Values are in source-image pixels and intentionally do not affect formal
# Step 7 projection evidence or Scene-Fact geometry.
REVIEW_PRINCIPAL_POINT_Y_OFFSETS_BY_CLIP = {
    "test_clip_894": {
        "front_wide": -16.0,
        "front_tele": -35.0,
    },
}


def review_principal_point_y_offset(clip_id, camera_name):
    return REVIEW_PRINCIPAL_POINT_Y_OFFSETS_BY_CLIP.get(
        str(clip_id),
        {},
    ).get(str(camera_name), 0.0)


def align_review_calibration(calibration, *, clip_id):
    offset = review_principal_point_y_offset(
        clip_id,
        calibration.camera_name,
    )
    if offset == 0.0:
        return calibration
    return replace(
        calibration,
        principal_point_y=calibration.principal_point_y + offset,
    )


def review_box_color(label):
    prefix = str(label).split(" ", 1)[0].rstrip("0123456789")
    return REVIEW_BOX_COLORS.get(prefix, (0, 0, 255))


def selected_actor_labels(row):
    labels = {}
    context = row.get("actor_context", {})
    for role, prefix in ROLE_LABEL_PREFIXES.items():
        for actor in context.get(role, ()):
            track_id = str(actor["track_id"])
            rank = int(actor.get("role_rank", len(labels) + 1))
            labels[track_id] = f"{prefix}{rank} track {track_id}"
    return labels


def _actor_records(value):
    """Yield Actor dictionaries from recorded snapshot containers."""
    if isinstance(value, Mapping):
        if all(key in value for key in ("track_id", "label_class", "pose", "dimensions")):
            yield value
            return
        for child in value.values():
            yield from _actor_records(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _actor_records(child)


def _camera_identity_tokens(camera_name):
    return tuple(part for part in str(camera_name).lower().split("_") if part)


def _calibration_identity_score(path, raw, camera_name):
    available = raw.get("available_camera")
    if not isinstance(available, Mapping):
        return None
    logical_id = str(available.get("logical_id", "")).lower()
    identity = f"{path.as_posix().lower()} {logical_id}"
    tokens = _camera_identity_tokens(camera_name)
    if not all(token in identity for token in tokens):
        return None
    normalized_camera = "".join(tokens)
    normalized_logical = "".join(character for character in logical_id if character.isalnum())
    normalized_path = "".join(character for character in path.as_posix().lower() if character.isalnum())
    exact_logical = int(normalized_camera == normalized_logical)
    logical_contains = int(normalized_camera in normalized_logical)
    path_contains = int(normalized_camera in normalized_path)
    calibration_hint = int("calib" in path.as_posix().lower())
    return (
        exact_logical,
        logical_contains,
        path_contains,
        calibration_hint,
        -len(path.as_posix()),
    )


def _load_review_calibrations(clip_root, sample_paths):
    calibrations = {}
    json_paths = tuple(clip_root.rglob("*.json"))
    for camera_name, image_path in sample_paths.items():
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"unable to inspect camera image: {image_path}")
        height, width = image.shape[:2]
        matches = []
        for path in json_paths:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(raw, Mapping):
                continue
            score = _calibration_identity_score(path, raw, camera_name)
            if score is None:
                continue
            try:
                calibration = load_camera_calibration(
                    path,
                    camera_name=camera_name,
                    source_width=width,
                    source_height=height,
                )
            except (KeyError, TypeError, ValueError):
                continue
            matches.append((score, path, calibration))
        if not matches:
            raise RuntimeError(
                f"identity-matched camera calibration not found for {camera_name} under {clip_root}"
            )
        matches.sort(key=lambda value: (value[0], value[1].as_posix()), reverse=True)
        score, selected_path, calibration = matches[0]
        calibration = align_review_calibration(
            calibration,
            clip_id=clip_root.name,
        )
        calibrations[camera_name] = calibration
        print(
            "Review calibration:",
            camera_name,
            "path=",
            selected_path,
            "logical_id=",
            calibration.logical_id,
            "image=",
            f"{width}x{height}",
            "principal=",
            f"({calibration.principal_point_x:.3f}, {calibration.principal_point_y:.3f})",
            "review_y_offset_px=",
            review_principal_point_y_offset(
                clip_root.name,
                camera_name,
            ),
        )
    return calibrations

def _frame_actor_overlays(reader, stamp_ns, selected_labels, calibrations):
    actors = reader.get_actors_at(stamp_ns, tolerance_ns=CAMERA_TOLERANCE_NS)
    ego = reader.get_recorded_ego_state_at_or_before(
        stamp_ns,
        maximum_age_ns=CAMERA_TOLERANCE_NS,
    )
    if actors is None or ego is None:
        return {name: () for name in CAMERA_NAMES}
    by_track = {
        str(actor["track_id"]): actor
        for actor in _actor_records(actors.message)
    }
    overlays = {name: [] for name in CAMERA_NAMES}
    for track_id, label in selected_labels.items():
        actor = by_track.get(track_id)
        if actor is None:
            continue
        for camera_name, calibration in calibrations.items():
            projection = project_actor_box_to_camera(
                actor,
                recorded_ego_message=ego.message,
                calibration=calibration,
            )
            bbox = projection.clipped_bbox
            if not projection.projection_valid or bbox is None:
                continue
            overlays[camera_name].append((bbox, label))
    return {name: tuple(values) for name, values in overlays.items()}


def _draw_review_overlays(image, overlays):
    for bbox, label in overlays:
        color = review_box_color(label)
        x1 = max(0, min(image.shape[1] - 1, round(bbox.min_u)))
        y1 = max(0, min(image.shape[0] - 1, round(bbox.min_v)))
        x2 = max(0, min(image.shape[1] - 1, round(bbox.max_u)))
        y2 = max(0, min(image.shape[0] - 1, round(bbox.max_v)))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        label_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            2,
        )
        label_top = max(0, y1 - label_size[1] - baseline - 8)
        label_right = min(image.shape[1] - 1, x1 + label_size[0] + 10)
        cv2.rectangle(
            image,
            (x1, label_top),
            (label_right, y1),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 5, max(label_size[1] + 2, y1 - baseline - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return image



def labeled_camera_frame(path, camera_name, width, height, overlays=()):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"unable to read camera image: {path}")
    image = _draw_review_overlays(image, overlays)
    image = letterbox(image, width, height)
    cv2.rectangle(image, (0, 0), (width, 44), (0, 0, 0), -1)
    cv2.putText(
        image,
        CAMERA_LABELS[camera_name],
        (14, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return image


def compose_frame(
    paths,
    *,
    anchor_id,
    relative_seconds,
    panel_width,
    panel_height,
    overlays=None,
):
    overlays = overlays or {name: () for name in CAMERA_NAMES}
    panels = {
        name: labeled_camera_frame(
            paths[name],
            name,
            panel_width,
            panel_height,
            overlays.get(name, ()),
        )
        for name in CAMERA_NAMES
    }
    top = cv2.hconcat((panels["front_wide"], panels["front_tele"]))
    bottom = cv2.hconcat((panels["cross_left"], panels["cross_right"]))
    frame = cv2.vconcat((top, bottom))
    marker = "ANCHOR" if abs(relative_seconds) <= 0.06 else ""
    footer = f"{anchor_id}   t={relative_seconds:+.2f}s   {marker}".rstrip()
    cv2.rectangle(frame, (0, frame.shape[0] - 42), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    color = (0, 255, 255) if marker else (255, 255, 255)
    cv2.putText(frame, footer, (14, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)
    return frame


def create_review_video(
    row,
    *,
    dataset_root,
    output_path,
    pre_seconds,
    post_seconds,
    fps,
    panel_width,
    panel_height,
    force,
):
    if output_path.exists() and not force:
        raise FileExistsError(f"review video exists: {output_path}; use --force")
    clip_root = dataset_root / str(row["clip_id"])
    reader = DrivingClipReader(clip_root)
    anchor_ns = int(row["anchor_ns"])
    start_ns = anchor_ns - round(pre_seconds * 1_000_000_000)
    end_ns = anchor_ns + round(post_seconds * 1_000_000_000)
    stamps = target_timestamps(reader, start_ns=start_ns, end_ns=end_ns)
    sample_paths = synchronized_paths(reader, min(stamps, key=lambda value: abs(value - anchor_ns)))
    if sample_paths is None:
        raise RuntimeError("no complete four-camera sample exists for calibration scaling")
    calibrations = _load_review_calibrations(clip_root, sample_paths)
    selected_labels = selected_actor_labels(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (panel_width * 2, panel_height * 2),
    )
    if not writer.isOpened():
        raise RuntimeError(f"unable to open video writer: {output_path}")
    written = 0
    skipped = 0
    try:
        for stamp_ns in stamps:
            paths = synchronized_paths(reader, stamp_ns)
            if paths is None:
                skipped += 1
                continue
            overlays = _frame_actor_overlays(
                reader,
                stamp_ns,
                selected_labels,
                calibrations,
            )
            frame = compose_frame(
                paths,
                anchor_id=str(row["anchor_id"]),
                relative_seconds=(stamp_ns - anchor_ns) / 1_000_000_000,
                panel_width=panel_width,
                panel_height=panel_height,
                overlays=overlays,
            )
            writer.write(frame)
            written += 1
    finally:
        writer.release()
    if written < 2:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("fewer than two complete four-camera frame sets exist in the requested review window")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("review video was not created")
    if skipped:
        print("Incomplete synchronized timestamps skipped:", skipped)
    print("Actor overlay labels:", selected_labels)
    return written

if __name__ == "__main__":
    raise SystemExit(main())
