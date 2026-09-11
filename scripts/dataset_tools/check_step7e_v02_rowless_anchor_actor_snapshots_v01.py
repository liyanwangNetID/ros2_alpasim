#!/usr/bin/env python3
"""Validate exact Actor snapshots for Step 7E v02 rowless Anchors.

The script derives rowless Anchors from source Keyframes and the completed v02
Actor JSONL, then reads each exact Actor snapshot through DrivingClipReader. It
reports the Actor count for every rowless Anchor and classifies the reason
without rerunning projection, occlusion, or export geometry.
"""

from __future__ import annotations

import json
from collections import Counter

from actor_export_anchor_coverage_summary_v01 import (
    summarize_actor_export_anchor_coverage,
)
from step2.clip_reader import DrivingClipReader
from export_step7e_geometric_occlusion_evidence_v01 import read_keyframes
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT


KEYFRAMES = ANNOTATION_ROOT / "keyframes.jsonl"
INPUT = ANNOTATION_ROOT / "step7e_geometric_occlusion_evidence_v02.jsonl"
EXPECTED_ROWLESS_ANCHOR_COUNT = 26


def read_exported_identities() -> tuple[tuple[str, str], ...]:
    identities = []
    with INPUT.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for field in ("anchor_id", "track_id"):
                if field not in row:
                    raise ValueError(
                        f"{INPUT}:{line_number}: missing {field}"
                    )
            identities.append(
                (str(row["anchor_id"]), str(row["track_id"]))
            )
    return tuple(identities)


def inspect_exact_actor_snapshot(reader, anchor_ns: int):
    snapshots = reader.get_actor_snapshots(anchor_ns, duration_ns=0)
    if len(snapshots) != 1:
        return "exact_snapshot_count_not_one", None, len(snapshots)
    snapshot = snapshots[0]
    if snapshot.stamp_ns != anchor_ns:
        return "snapshot_timestamp_not_exact", None, snapshot.stamp_ns
    actors = snapshot.message.get("actors")
    if not isinstance(actors, list):
        return "actors_field_not_list", None, type(actors).__name__
    if actors:
        return "exact_snapshot_has_actors", len(actors), None
    return "exact_snapshot_empty_actor_list", 0, None


def main() -> int:
    keyframes = read_keyframes(KEYFRAMES)
    keyframes_by_id = {
        str(keyframe["anchor_id"]): keyframe for keyframe in keyframes
    }
    coverage = summarize_actor_export_anchor_coverage(
        keyframes=keyframes,
        exported_actor_identities=read_exported_identities(),
    )
    if coverage.anchor_without_actor_rows_count != EXPECTED_ROWLESS_ANCHOR_COUNT:
        raise RuntimeError(
            "Rowless Anchor baseline changed: "
            f"actual={coverage.anchor_without_actor_rows_count}, "
            f"expected={EXPECTED_ROWLESS_ANCHOR_COUNT}"
        )

    results = []
    reason_counts = Counter()
    current_clip_id = None
    reader = None

    for anchor_id in coverage.anchor_without_actor_rows:
        keyframe = keyframes_by_id[anchor_id]
        clip_id = str(keyframe["clip_id"])
        anchor_ns = int(keyframe["anchor_ns"])
        if clip_id != current_clip_id:
            current_clip_id = clip_id
            reader = DrivingClipReader(ALPASIM_DATA_ROOT / clip_id)
        assert reader is not None

        reason, actor_count, detail = inspect_exact_actor_snapshot(
            reader,
            anchor_ns,
        )
        reason_counts[reason] += 1
        results.append(
            {
                "anchor_id": anchor_id,
                "clip_id": clip_id,
                "anchor_ns": anchor_ns,
                "reason": reason,
                "actor_count": actor_count,
                "detail": detail,
            }
        )

    print("Step 7E v02 rowless-Anchor Actor snapshot inspection")
    print("rowless_anchor_count:", len(results))
    print("reason_counts:", dict(sorted(reason_counts.items())))
    print("rows:")
    for result in results:
        print(
            f"  anchor={result['anchor_id']} "
            f"reason={result['reason']} "
            f"actor_count={result['actor_count']} "
            f"detail={result['detail']}"
        )
    print("keyframes:", KEYFRAMES)
    print("input:", INPUT)

    unexpected = [
        result
        for result in results
        if result["reason"] != "exact_snapshot_empty_actor_list"
    ]
    if unexpected:
        raise RuntimeError(
            f"{len(unexpected)} rowless Anchors are not explained by an "
            "exact empty Actor list"
        )

    print(
        "PASS: every rowless Anchor has one exact Actor snapshot with an "
        "empty actors list."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
