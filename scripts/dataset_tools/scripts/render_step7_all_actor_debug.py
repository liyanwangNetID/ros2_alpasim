#!/usr/bin/env python3
"""Render exact-Anchor camera images with every valid Actor projection and track id."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from step2.clip_reader import DrivingClipReader
from step7.projection import project_actor_box_to_camera
from step7.review_scene_fact import (
    CAMERA_NAMES,
    _actor_records,
    _load_review_calibrations,
    synchronized_paths,
)

ROLE_COLORS = {
    "lead_actors": (0, 0, 255),
    "left_nearby_actors": (0, 255, 255),
    "right_nearby_actors": (0, 255, 0),
}
DEFAULT_COLOR = (255, 255, 0)


def load_scene_fact(path, anchor_id):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row["anchor_id"]) == anchor_id:
                return row
    raise KeyError(f"Scene-Fact Anchor not found: {anchor_id}")


def selected_styles(row):
    styles = {}
    context = row["actor_context"]
    for role, color in ROLE_COLORS.items():
        for actor in context[role]:
            styles[str(actor["track_id"])] = (
                color,
                f"{role}:{actor['role_rank']} track {actor['track_id']}",
                4,
            )
    return styles


def draw_box(image, bbox, label, color, thickness):
    x1 = max(0, min(image.shape[1] - 1, round(bbox.min_u)))
    y1 = max(0, min(image.shape[0] - 1, round(bbox.min_v)))
    x2 = max(0, min(image.shape[1] - 1, round(bbox.max_u)))
    y2 = max(0, min(image.shape[0] - 1, round(bbox.max_v)))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    y = max(18, y1 - 5)
    cv2.putText(image, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-id", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--scene-facts", type=Path)
    args = parser.parse_args()

    dataset_root = Path(os.environ["ALPASIM_DATA_ROOT"])
    scene_facts = args.scene_facts or dataset_root / "annotations/v0.1-draft/scene_facts.jsonl"
    row = load_scene_fact(scene_facts, args.anchor_id)
    clip_root = dataset_root / str(row["clip_id"])
    anchor_ns = int(row["anchor_ns"])
    reader = DrivingClipReader(clip_root)
    actor_result = reader.get_actors_at(anchor_ns, tolerance_ns=0)
    ego_result = reader.get_recorded_ego_state_at_or_before(anchor_ns, maximum_age_ns=0)
    paths = synchronized_paths(reader, anchor_ns)
    if actor_result is None or ego_result is None or paths is None:
        raise RuntimeError("Exact Anchor Actor, Ego, or camera data is unavailable")
    calibrations = _load_review_calibrations(clip_root, paths)
    actors = tuple(_actor_records(actor_result.message))
    styles = selected_styles(row)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    for camera_name in CAMERA_NAMES:
        image = cv2.imread(str(paths[camera_name]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unable to read image: {paths[camera_name]}")
        rendered = 0
        for actor in actors:
            projection = project_actor_box_to_camera(
                actor,
                recorded_ego_message=ego_result.message,
                calibration=calibrations[camera_name],
            )
            if not projection.projection_valid or projection.clipped_bbox is None:
                continue
            track_id = str(actor["track_id"])
            actor_class = str(actor["label_class"])
            color, label, thickness = styles.get(
                track_id,
                (DEFAULT_COLOR, f"track {track_id} {actor_class}", 1),
            )
            draw_box(image, projection.clipped_bbox, label, color, thickness)
            rendered += 1
        output = args.output_directory / f"{args.anchor_id}_{camera_name}_all_actors.jpg"
        if not cv2.imwrite(str(output), image):
            raise RuntimeError(f"Unable to write image: {output}")
        print(camera_name, "boxes:", rendered, "output:", output)


if __name__ == "__main__":
    main()
