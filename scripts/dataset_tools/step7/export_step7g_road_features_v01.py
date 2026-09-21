#!/usr/bin/env python3
"""Export indexed Step 7G road and wait-line features with progress and ETA."""
from __future__ import annotations

import hashlib
import json
import os
import time

from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT, INTERMEDIATE_ROOT
from step2.clip_reader import DrivingClipReader
from step7.road_lane_features_summary_v01 import summarize_road_feature_rows
from step7.road_lane_features_v01 import (
    RoadFeatureMapContext,
    compute_ego_and_actor_road_features,
)
from step7.scene_fact_geometry_v01 import recorded_ego_pose_and_speed

KEYFRAMES = ANNOTATION_ROOT / "keyframes.jsonl"
ACTOR_OUTPUT = INTERMEDIATE_ROOT / "actor_road_features_v0.1.jsonl"
EGO_OUTPUT = INTERMEDIATE_ROOT / "ego_road_features_v0.1.jsonl"
SUMMARY = ANNOTATION_ROOT / "step7g_road_features_summary_v01.json"
PROGRESS_INTERVAL_S = 10.0
PROGRESS_INTERVAL_KEYFRAMES = 25


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def compact(row):
    return json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def duration_text(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def print_progress(*, completed, total, actor_rows, started, stage="matching"):
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed > 0.0 else 0.0
    eta = (total - completed) / rate if rate > 0.0 else 0.0
    percent = 100.0 * completed / total if total else 100.0
    print(
        f"[Step 7G:{stage}] {completed}/{total} Keyframes | "
        f"{percent:5.1f}% | Actor rows {actor_rows} | "
        f"elapsed {duration_text(elapsed)} | rate {rate:.2f} Keyframes/s | "
        f"ETA {duration_text(eta)}",
        flush=True,
    )


def main():
    keyframes = sorted(
        read_jsonl(KEYFRAMES),
        key=lambda row: (
            str(row["clip_id"]), int(row["anchor_ns"]), str(row["anchor_id"])
        ),
    )
    readers = {}
    contexts = {}
    actor_rows = []
    ego_rows = []
    actor_tmp = ACTOR_OUTPUT.with_suffix(".jsonl.tmp")
    ego_tmp = EGO_OUTPUT.with_suffix(".jsonl.tmp")
    INTERMEDIATE_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_progress = started
    print_progress(
        completed=0,
        total=len(keyframes),
        actor_rows=0,
        started=started,
        stage="starting",
    )
    with actor_tmp.open("w", encoding="utf-8") as actor_file, ego_tmp.open(
        "w", encoding="utf-8"
    ) as ego_file:
        for index, keyframe in enumerate(keyframes, start=1):
            clip_id = str(keyframe["clip_id"])
            anchor_id = str(keyframe["anchor_id"])
            anchor_ns = int(keyframe["anchor_ns"])
            if clip_id not in readers:
                reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
                readers[clip_id] = reader
                contexts[clip_id] = RoadFeatureMapContext(
                    raw_map=reader.get_vector_map()
                )
            reader = readers[clip_id]
            context = contexts[clip_id]
            ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
            actors = reader.get_actors_at(anchor_ns)
            if (
                ego is None
                or ego.stamp_ns != anchor_ns
                or actors is None
                or actors.stamp_ns != anchor_ns
            ):
                raise RuntimeError(f"exact current data unavailable: {anchor_id}")
            ego_pose, _ = recorded_ego_pose_and_speed(ego.message)
            ego_match, actor_matches = compute_ego_and_actor_road_features(
                context=context,
                ego_pose=ego_pose,
                actors=actors.message["actors"],
            )
            ego_row = {
                "schema_version": "step7g-ego-road-features-v01",
                "anchor_id": anchor_id,
                "clip_id": clip_id,
                "anchor_ns": anchor_ns,
                "lane_match_status": ego_match.status,
                **{
                    key: value
                    for key, value in ego_match.to_dict().items()
                    if key != "status"
                },
            }
            ego_file.write(compact(ego_row))
            ego_rows.append(ego_row)
            for track_id, label, match, relation in actor_matches:
                row = {
                    "schema_version": "step7g-actor-road-features-v01",
                    "anchor_id": anchor_id,
                    "clip_id": clip_id,
                    "anchor_ns": anchor_ns,
                    "track_id": track_id,
                    "label_class": label,
                    "lane_match_status": match.status,
                    "ego_lane_id": ego_match.lane_id,
                    "ego_lane_relation": relation,
                    **{
                        key: value
                        for key, value in match.to_dict().items()
                        if key != "status"
                    },
                }
                actor_file.write(compact(row))
                actor_rows.append(row)
            now = time.monotonic()
            if (
                index == len(keyframes)
                or index % PROGRESS_INTERVAL_KEYFRAMES == 0
                or now - last_progress >= PROGRESS_INTERVAL_S
            ):
                print_progress(
                    completed=index,
                    total=len(keyframes),
                    actor_rows=len(actor_rows),
                    started=started,
                )
                last_progress = now
    os.replace(actor_tmp, ACTOR_OUTPUT)
    os.replace(ego_tmp, EGO_OUTPUT)
    print("[Step 7G:summary] validating coverage and hashing outputs", flush=True)
    report = summarize_road_feature_rows(
        keyframes=keyframes,
        actor_rows=actor_rows,
        ego_rows=ego_rows,
    )
    report.update(
        {
            "actor_output_path": str(ACTOR_OUTPUT),
            "ego_output_path": str(EGO_OUTPUT),
            "actor_output_sha256": digest(ACTOR_OUTPUT),
            "ego_output_sha256": digest(EGO_OUTPUT),
            "future_data_used": False,
            "spatial_index_used": True,
            "spatial_cell_size_m": 20.0,
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    summary_tmp = SUMMARY.with_suffix(".json.tmp")
    summary_tmp.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(summary_tmp, SUMMARY)
    print("Step 7G indexed road-feature export")
    print("keyframes:", report["keyframe_count"])
    print("actor rows:", report["actor_row_count"])
    print("ego match statuses:", report["ego_match_status_counts"])
    print("actor match statuses:", report["actor_match_status_counts"])
    print("actor/ego lane relations:", report["actor_ego_lane_relation_counts"])
    print("actor sha256:", report["actor_output_sha256"])
    print("ego sha256:", report["ego_output_sha256"])
    print("elapsed:", duration_text(report["elapsed_seconds"]))
    print("summary:", SUMMARY)
    print("PASS: indexed Step 7G exported with progress and current data only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
