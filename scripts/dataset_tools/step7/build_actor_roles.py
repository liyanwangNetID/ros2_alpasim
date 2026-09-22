#!/usr/bin/env python3
"""Build Step 7H deterministic Actor-role selections with progress and ETA."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict

from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT, INTERMEDIATE_ROOT
from step2.clip_reader import DrivingClipReader
from step7.observability import (
    evaluate_frozen_dynamic_occlusion_policy,
)
from step7.actor_roles import summarize_actor_role_rows
from step7.actor_roles import (
    empty_role_reasons,
    join_actor_role_inputs,
    select_actor_roles,
)
from step7.scene_facts import compute_snapshot_actor_geometries

KEYFRAMES = ANNOTATION_ROOT / "keyframes.jsonl"
EVIDENCE = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.jsonl"
HISTORY = INTERMEDIATE_ROOT / "actor_short_history_features_v0.1.jsonl"
ROAD = INTERMEDIATE_ROOT / "actor_road_features_v0.1.jsonl"
OUTPUT = INTERMEDIATE_ROOT / "actor_role_selection_v0.1.jsonl"
SUMMARY = ANNOTATION_ROOT / "step7h_actor_role_selection_summary_v01.json"
PROGRESS_INTERVAL_S = 10.0
PROGRESS_INTERVAL_KEYFRAMES = 100


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def grouped(rows):
    result = defaultdict(list)
    for row in rows:
        result[str(row["anchor_id"])].append(row)
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def compact(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def duration_text(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress(completed, total, assignments, started):
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    eta = (total - completed) / rate if rate else 0.0
    print(
        f"[Step 7H] {completed}/{total} Keyframes | {100.0 * completed / total:5.1f}% | "
        f"assignments {assignments} | elapsed {duration_text(elapsed)} | "
        f"rate {rate:.2f} Keyframes/s | ETA {duration_text(eta)}",
        flush=True,
    )


def main():
    started = time.monotonic()
    keyframes = sorted(
        read_jsonl(KEYFRAMES),
        key=lambda row: (str(row["clip_id"]), int(row["anchor_ns"]), str(row["anchor_id"])),
    )
    print("[Step 7H:load] reading and indexing 7E, 7F, and 7G inputs", flush=True)
    evidence_rows = read_jsonl(EVIDENCE)
    history_by_anchor = grouped(read_jsonl(HISTORY))
    road_by_anchor = grouped(read_jsonl(ROAD))
    visibility_decisions = evaluate_frozen_dynamic_occlusion_policy(
        evidence_rows=evidence_rows
    )
    visibility_by_anchor = defaultdict(list)
    evidence_by_identity = {
        (str(row["anchor_id"]), str(row["track_id"])): row
        for row in evidence_rows
    }
    for decision in visibility_decisions:
        source = evidence_by_identity[(decision.anchor_id, decision.track_id)]
        visibility_by_anchor[decision.anchor_id].append({
            "anchor_id": decision.anchor_id,
            "track_id": decision.track_id,
            "label_class": str(source["label_class"]),
            "shadow_status": decision.shadow_status,
            "winning_cell_count": decision.winning_cell_count,
            "maximum_visible_fraction": decision.maximum_visible_fraction,
        })
    readers = {}
    rows = []
    temporary = OUTPUT.with_suffix(".jsonl.tmp")
    last_progress = started
    assignment_count = 0
    with temporary.open("w", encoding="utf-8") as output:
        for index, keyframe in enumerate(keyframes, start=1):
            anchor_id = str(keyframe["anchor_id"])
            clip_id = str(keyframe["clip_id"])
            anchor_ns = int(keyframe["anchor_ns"])
            reader = readers.setdefault(
                clip_id, DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
            )
            ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
            actors = reader.get_actors_at(anchor_ns)
            if ego is None or ego.stamp_ns != anchor_ns or actors is None or actors.stamp_ns != anchor_ns:
                raise RuntimeError(f"exact current data unavailable: {anchor_id}")
            geometry_rows = []
            for feature in compute_snapshot_actor_geometries(
                actors.message,
                recorded_ego_message=ego.message,
            ):
                geometry_rows.append({
                    "anchor_id": anchor_id,
                    "track_id": feature.track_id,
                    "label_class": feature.actor_class,
                    **feature.to_dict(),
                })
            candidates = join_actor_role_inputs(
                geometry_rows=geometry_rows,
                visibility_rows=visibility_by_anchor.get(anchor_id, ()),
                history_rows=history_by_anchor.get(anchor_id, ()),
                road_rows=road_by_anchor.get(anchor_id, ()),
            )
            roles = select_actor_roles(candidates=candidates)
            empty_reasons = empty_role_reasons(candidates=candidates, roles=roles)
            assignment_count += sum(value is not None for value in roles.values())
            row = {
                "schema_version": "step7h-actor-role-selection-v01",
                "anchor_id": anchor_id,
                "clip_id": clip_id,
                "anchor_ns": anchor_ns,
                "candidate_actor_count": len(candidates),
                "roles": roles,
                "empty_role_reasons": empty_reasons,
                "future_actor_data_used": False,
                "future_ego_data_used": False,
                "planner_output_used": False,
                "meta_action_used": False,
            }
            output.write(compact(row))
            rows.append(row)
            now = time.monotonic()
            if index == len(keyframes) or index % PROGRESS_INTERVAL_KEYFRAMES == 0 or now - last_progress >= PROGRESS_INTERVAL_S:
                progress(index, len(keyframes), assignment_count, started)
                last_progress = now
    os.replace(temporary, OUTPUT)
    report = summarize_actor_role_rows(keyframes=keyframes, rows=rows)
    report.update({
        "output_path": str(OUTPUT),
        "output_sha256": digest(OUTPUT),
        "elapsed_seconds": time.monotonic() - started,
        "source_evidence_path": str(EVIDENCE),
        "source_history_path": str(HISTORY),
        "source_road_path": str(ROAD),
        "geometry_computed_from_current_snapshots": True,
        "future_data_used": False,
    })
    temporary_summary = SUMMARY.with_suffix(".json.tmp")
    temporary_summary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_summary, SUMMARY)
    print("Step 7H Actor-role selection export")
    print("keyframes:", report["keyframe_count"])
    print("selected roles:", report["selected_role_counts"])
    print("empty reasons:", report["empty_role_reason_counts"])
    print("role conflicts:", report["role_conflict_count"])
    print("sha256:", report["output_sha256"])
    print("elapsed:", duration_text(report["elapsed_seconds"]))
    print("output:", OUTPUT)
    print("summary:", SUMMARY)
    print("PASS: Step 7H exported deterministically from current and past evidence only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
