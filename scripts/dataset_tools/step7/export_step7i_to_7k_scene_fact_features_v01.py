#!/usr/bin/env python3
"""Export Step 7I-7K Scene-Fact feature rows with progress and ETA."""
from __future__ import annotations
import hashlib, json, os, time
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT, INTERMEDIATE_ROOT
from step2.clip_reader import DrivingClipReader
from step7.scene_fact_features_v01 import assemble_scene_fact_feature_row
from step7.scene_fact_feature_summary_v01 import summarize_scene_fact_feature_rows
from step7.scene_fact_geometry_v01 import compute_snapshot_actor_geometries

KEYFRAMES = ANNOTATION_ROOT / "keyframes.jsonl"
EGO_ROAD = INTERMEDIATE_ROOT / "ego_road_features_v0.1.jsonl"
ROLES = INTERMEDIATE_ROOT / "actor_role_selection_v0.1.jsonl"
OUTPUT = INTERMEDIATE_ROOT / "scene_fact_features_v0.1.jsonl"
SUMMARY = ANNOTATION_ROOT / "step7k_scene_fact_features_summary_v01.json"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def unique_by_anchor(rows, name):
    result = {}
    for row in rows:
        anchor = str(row["anchor_id"])
        if anchor in result:
            raise ValueError(f"duplicate {name} Anchor: {anchor}")
        result[anchor] = row
    return result


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def duration_text(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def progress(completed, total, started):
    elapsed = time.monotonic() - started
    rate = completed / elapsed if elapsed else 0.0
    eta = (total - completed) / rate if rate else 0.0
    print(
        f"[Step 7I-7K] {completed}/{total} Keyframes | {100.0*completed/total:5.1f}% | "
        f"elapsed {duration_text(elapsed)} | rate {rate:.2f} Keyframes/s | ETA {duration_text(eta)}",
        flush=True,
    )


def main():
    started = time.monotonic()
    print("[Step 7I-7K:load] reading Keyframes, Ego road, and role selections", flush=True)
    keyframes = sorted(read_jsonl(KEYFRAMES), key=lambda row: (
        str(row["clip_id"]), int(row["anchor_ns"]), str(row["anchor_id"])
    ))
    ego_road = unique_by_anchor(read_jsonl(EGO_ROAD), "Ego road")
    roles = unique_by_anchor(read_jsonl(ROLES), "role selection")
    expected = {str(row["anchor_id"]) for row in keyframes}
    if set(ego_road) != expected or set(roles) != expected:
        raise ValueError("7I-7K Anchor inputs do not close to Keyframes")
    readers = {}
    rows = []
    temporary = OUTPUT.with_suffix(".jsonl.tmp")
    last_progress = started
    with temporary.open("w", encoding="utf-8") as output:
        for index, keyframe in enumerate(keyframes, start=1):
            anchor = str(keyframe["anchor_id"])
            clip = str(keyframe["clip_id"])
            anchor_ns = int(keyframe["anchor_ns"])
            reader = readers.setdefault(clip, DrivingClipReader(ALPASIM_DATA_ROOT / clip))
            ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
            actors = reader.get_actors_at(anchor_ns)
            if ego is None or ego.stamp_ns != anchor_ns or actors is None or actors.stamp_ns != anchor_ns:
                raise RuntimeError(f"exact current data unavailable: {anchor}")
            geometries = {
                feature.track_id: {"track_id": feature.track_id, **feature.to_dict()}
                for feature in compute_snapshot_actor_geometries(
                    actors.message, recorded_ego_message=ego.message
                )
            }
            row = assemble_scene_fact_feature_row(
                keyframe=keyframe,
                ego_road=ego_road[anchor],
                role_selection=roles[anchor],
                current_geometry_by_track_id=geometries,
            )
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            rows.append(row)
            now = time.monotonic()
            if index == len(keyframes) or index % 100 == 0 or now - last_progress >= 10.0:
                progress(index, len(keyframes), started)
                last_progress = now
    os.replace(temporary, OUTPUT)
    report = summarize_scene_fact_feature_rows(keyframes=keyframes, rows=rows)
    report.update({
        "output_path": str(OUTPUT),
        "output_sha256": digest(OUTPUT),
        "elapsed_seconds": time.monotonic() - started,
        "source_ego_road_path": str(EGO_ROAD),
        "source_role_selection_path": str(ROLES),
        "future_data_used": False,
    })
    temporary_summary = SUMMARY.with_suffix(".json.tmp")
    temporary_summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_summary, SUMMARY)
    print("Step 7I-7K Scene-Fact feature export")
    print("keyframes:", report["keyframe_count"])
    print("road contexts:", report["road_context_type_counts"])
    print("quality statuses:", report["quality_status_counts"])
    print("role presence:", report["role_presence_status_counts"])
    print("sha256:", report["output_sha256"])
    print("elapsed:", duration_text(report["elapsed_seconds"]))
    print("output:", OUTPUT)
    print("summary:", SUMMARY)
    print("PASS: Step 7I-7K features exported from current and past evidence only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
