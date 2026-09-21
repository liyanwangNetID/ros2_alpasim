"""Step 7F short-history export summary."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


def summarize_actor_short_history_rows(
    *, keyframes: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    anchor_ids = [str(item["anchor_id"]) for item in keyframes]
    if len(anchor_ids) != len(set(anchor_ids)):
        raise ValueError("keyframe anchor_id values must be unique")
    identities = [(str(row["anchor_id"]), str(row["track_id"])) for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("history row identities must be unique")
    unknown = sorted({anchor for anchor, _ in identities} - set(anchor_ids))
    if unknown:
        raise ValueError("history rows contain unknown anchor_id values")
    status_counts = Counter(str(row["history_status"]) for row in rows)
    class_counts = Counter(str(row["label_class"]) for row in rows)
    exported = {anchor for anchor, _ in identities}
    return {
        "schema_version": "step7f-actor-short-history-summary-v01",
        "keyframe_count": len(anchor_ids),
        "actor_row_count": len(rows),
        "anchor_with_actor_rows_count": len(exported),
        "anchor_without_actor_rows_count": len(set(anchor_ids) - exported),
        "history_status_counts": dict(sorted(status_counts.items())),
        "actor_class_counts": dict(sorted(class_counts.items())),
    }
