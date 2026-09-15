from types import SimpleNamespace

from step7.actor_export_anchor_snapshot_coverage_v01 import (
    summarize_actor_export_anchor_snapshot_coverage,
)


def snapshot(stamp_ns, actors):
    return SimpleNamespace(
        stamp_ns=stamp_ns,
        message={"actors": actors},
    )


def test_classifies_empty_rowless_anchors_and_reuses_reader_per_clip():
    created = []

    def factory(clip_id):
        created.append(clip_id)
        return SimpleNamespace(
            get_actor_snapshots=lambda anchor_ns, duration_ns: (
                snapshot(anchor_ns, []),
            )
        )

    result = summarize_actor_export_anchor_snapshot_coverage(
        keyframes=(
            {"anchor_id": "a1", "clip_id": "a", "anchor_ns": 1},
            {"anchor_id": "a2", "clip_id": "a", "anchor_ns": 2},
            {"anchor_id": "b1", "clip_id": "b", "anchor_ns": 3},
        ),
        exported_actor_identities=(("b1", "9"),),
        reader_factory=factory,
    )

    assert result.coverage.source_keyframe_count == 3
    assert result.coverage.anchor_without_actor_rows == ("a1", "a2")
    assert result.rowless_anchor_reason_counts == (
        ("exact_snapshot_empty_actor_list", 2),
    )
    assert created == ["a"]
    assert all(
        item.actor_count == 0
        for item in result.rowless_anchor_snapshot_evidence
    )


def test_reports_rowless_anchor_that_still_has_actors():
    reader = SimpleNamespace(
        get_actor_snapshots=lambda anchor_ns, duration_ns: (
            snapshot(anchor_ns, [{"track_id": "1"}]),
        )
    )
    result = summarize_actor_export_anchor_snapshot_coverage(
        keyframes=(
            {"anchor_id": "a", "clip_id": "clip", "anchor_ns": 1},
        ),
        exported_actor_identities=(),
        reader_factory=lambda clip_id: reader,
    )

    assert result.rowless_anchor_reason_counts == (
        ("exact_snapshot_has_actors", 1),
    )
    assert result.rowless_anchor_snapshot_evidence[0].actor_count == 1


def test_no_rowless_anchors_does_not_create_reader():
    calls = []
    result = summarize_actor_export_anchor_snapshot_coverage(
        keyframes=(
            {"anchor_id": "a", "clip_id": "clip", "anchor_ns": 1},
        ),
        exported_actor_identities=(("a", "1"),),
        reader_factory=lambda clip_id: calls.append(clip_id),
    )

    assert result.coverage.anchor_without_actor_rows_count == 0
    assert result.rowless_anchor_reason_counts == ()
    assert result.rowless_anchor_snapshot_evidence == ()
    assert calls == []


def test_to_dict_flattens_coverage_and_serializes_evidence():
    reader = SimpleNamespace(
        get_actor_snapshots=lambda anchor_ns, duration_ns: (
            snapshot(anchor_ns, []),
        )
    )
    result = summarize_actor_export_anchor_snapshot_coverage(
        keyframes=(
            {"anchor_id": "a", "clip_id": "clip", "anchor_ns": 1},
        ),
        exported_actor_identities=(),
        reader_factory=lambda clip_id: reader,
    ).to_dict()

    assert result["source_keyframe_count"] == 1
    assert result["rowless_anchor_reason_counts"] == {
        "exact_snapshot_empty_actor_list": 1
    }
    assert result["rowless_anchor_snapshot_evidence"][0]["anchor_id"] == "a"
