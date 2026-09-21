#!/usr/bin/env python3
"""Export compact Step 7F Actor short-history motion features."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT, INTERMEDIATE_ROOT
from step2.clip_reader import DrivingClipReader
from step7.actor_short_history_export_v01 import (
    actor_short_history_export_row,
    validate_actor_short_history_export_row,
)
from step7.actor_short_history_summary_v01 import summarize_actor_short_history_rows
from step7.actor_short_history_v01 import (
    DEFAULT_HISTORY_DURATION_NS,
    compute_snapshot_actor_short_histories,
)
from step7.scene_fact_geometry_v01 import recorded_ego_pose_and_speed

KEYFRAMES_PATH = ANNOTATION_ROOT / "keyframes.jsonl"
OUTPUT_PATH = INTERMEDIATE_ROOT / "actor_short_history_features_v0.1.jsonl"
SUMMARY_PATH = ANNOTATION_ROOT / "step7f_actor_short_history_summary_v01.json"


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    keyframes = sorted(
        read_jsonl(KEYFRAMES_PATH),
        key=lambda row: (
            str(row["clip_id"]), int(row["anchor_ns"]), str(row["anchor_id"])
        ),
    )
    if len({str(row["anchor_id"]) for row in keyframes}) != len(keyframes):
        raise RuntimeError("Keyframe anchor_id values must be unique")

    INTERMEDIATE_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_suffix(".jsonl.tmp")
    readers = {}
    exported_rows = []
    with temporary.open("w", encoding="utf-8") as output:
        for keyframe in keyframes:
            clip_id = str(keyframe["clip_id"])
            anchor_id = str(keyframe["anchor_id"])
            anchor_ns = int(keyframe["anchor_ns"])
            reader = readers.setdefault(
                clip_id, DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
            )
            ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
            if ego is None or ego.stamp_ns != anchor_ns:
                raise RuntimeError(f"Exact recorded Ego unavailable: {anchor_id}")
            anchor_pose, _ = recorded_ego_pose_and_speed(ego.message)
            snapshots = reader.get_actor_snapshots(
                anchor_ns, duration_ns=DEFAULT_HISTORY_DURATION_NS
            )
            if not snapshots or snapshots[-1].stamp_ns != anchor_ns:
                raise RuntimeError(f"Exact Anchor Actor snapshot unavailable: {anchor_id}")
            features = compute_snapshot_actor_short_histories(
                anchor_ns=anchor_ns,
                anchor_ego_pose=anchor_pose,
                snapshots=snapshots,
            )
            for feature in features:
                row = actor_short_history_export_row(
                    keyframe=keyframe,
                    feature=feature,
                    history_duration_ns=DEFAULT_HISTORY_DURATION_NS,
                )
                validate_actor_short_history_export_row(row)
                output.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ) + "\n"
                )
                exported_rows.append(row)
    os.replace(temporary, OUTPUT_PATH)

    summary = summarize_actor_short_history_rows(
        keyframes=keyframes, rows=exported_rows
    )
    summary.update({
        "source_keyframes_path": str(KEYFRAMES_PATH),
        "output_path": str(OUTPUT_PATH),
        "history_duration_ns": DEFAULT_HISTORY_DURATION_NS,
        "output_sha256": sha256(OUTPUT_PATH),
        "output_size_bytes": OUTPUT_PATH.stat().st_size,
        "raw_history_points_embedded": False,
        "future_actor_data_used": False,
        "future_ego_data_used": False,
    })
    temporary_summary = SUMMARY_PATH.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_summary, SUMMARY_PATH)

    print("Step 7F compact Actor short-history export")
    print("keyframes:", summary["keyframe_count"])
    print("actor rows:", summary["actor_row_count"])
    print("anchors with Actor rows:", summary["anchor_with_actor_rows_count"])
    print("anchors without Actor rows:", summary["anchor_without_actor_rows_count"])
    print("history statuses:", summary["history_status_counts"])
    print("raw history points embedded:", False)
    print("output size bytes:", summary["output_size_bytes"])
    print("output:", OUTPUT_PATH)
    print("sha256:", summary["output_sha256"])
    print("summary:", SUMMARY_PATH)
    print("PASS: compact Step 7F features exported from current and past data only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
