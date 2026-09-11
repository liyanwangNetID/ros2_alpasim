"""Summarize source-Keyframe coverage of Step 7E Actor export rows.

This module compares the complete source Keyframe collection with exported
Anchor/Actor identities. It distinguishes source Keyframes, Anchors with at
least one Actor row, and Anchors without Actor rows. It does not infer why an
Anchor has no Actor rows and does not recompute geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ActorExportAnchorCoverageSummary:
    source_keyframe_count: int
    anchor_with_actor_rows_count: int
    anchor_without_actor_rows_count: int
    anchor_without_actor_rows: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_keyframe_count": self.source_keyframe_count,
            "anchor_with_actor_rows_count": self.anchor_with_actor_rows_count,
            "anchor_without_actor_rows_count": self.anchor_without_actor_rows_count,
            "anchor_without_actor_rows": list(self.anchor_without_actor_rows),
        }


def summarize_actor_export_anchor_coverage(
    *,
    keyframes: Sequence[Mapping[str, Any]],
    exported_actor_identities: Sequence[tuple[str, str]],
) -> ActorExportAnchorCoverageSummary:
    """Compare source Anchor IDs with exported Anchor/Actor identities."""

    source = tuple(keyframes)
    source_anchor_ids = []
    for index, keyframe in enumerate(source):
        if "anchor_id" not in keyframe:
            raise ValueError(f"keyframe at index {index} is missing anchor_id")
        anchor_id = str(keyframe["anchor_id"])
        if not anchor_id:
            raise ValueError(f"keyframe at index {index} has empty anchor_id")
        source_anchor_ids.append(anchor_id)

    if len(set(source_anchor_ids)) != len(source_anchor_ids):
        raise ValueError("source keyframe anchor_id values must be unique")

    identities = tuple(
        (str(anchor_id), str(track_id))
        for anchor_id, track_id in exported_actor_identities
    )
    if any(not anchor_id or not track_id for anchor_id, track_id in identities):
        raise ValueError("exported Actor identities must be non-empty")
    if len(set(identities)) != len(identities):
        raise ValueError("exported Anchor/Actor identities must be unique")

    source_anchor_set = set(source_anchor_ids)
    exported_anchor_set = {anchor_id for anchor_id, _ in identities}
    unknown_exported_anchors = sorted(exported_anchor_set - source_anchor_set)
    if unknown_exported_anchors:
        raise ValueError(
            "exported rows contain Anchors absent from source Keyframes: "
            f"{unknown_exported_anchors}"
        )

    anchors_without_rows = tuple(
        sorted(source_anchor_set - exported_anchor_set)
    )
    anchors_with_rows = source_anchor_set & exported_anchor_set
    if len(anchors_with_rows) + len(anchors_without_rows) != len(source):
        raise RuntimeError("source Anchor coverage counts do not close")

    return ActorExportAnchorCoverageSummary(
        source_keyframe_count=len(source),
        anchor_with_actor_rows_count=len(anchors_with_rows),
        anchor_without_actor_rows_count=len(anchors_without_rows),
        anchor_without_actor_rows=anchors_without_rows,
    )
