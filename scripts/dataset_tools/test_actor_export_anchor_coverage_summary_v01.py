import pytest

from actor_export_anchor_coverage_summary_v01 import (
    summarize_actor_export_anchor_coverage,
)


def keyframe(anchor_id):
    return {"anchor_id": anchor_id}


def test_separates_source_anchors_with_and_without_actor_rows():
    result = summarize_actor_export_anchor_coverage(
        keyframes=(keyframe("c"), keyframe("a"), keyframe("b")),
        exported_actor_identities=(("a", "1"), ("a", "2"), ("c", "3")),
    )

    assert result.source_keyframe_count == 3
    assert result.anchor_with_actor_rows_count == 2
    assert result.anchor_without_actor_rows_count == 1
    assert result.anchor_without_actor_rows == ("b",)
    assert result.to_dict()["anchor_without_actor_rows"] == ["b"]


def test_all_source_anchors_may_have_rows():
    result = summarize_actor_export_anchor_coverage(
        keyframes=(keyframe("a"), keyframe("b")),
        exported_actor_identities=(("a", "1"), ("b", "2")),
    )

    assert result.anchor_without_actor_rows_count == 0
    assert result.anchor_without_actor_rows == ()


def test_empty_export_marks_every_source_anchor_without_rows():
    result = summarize_actor_export_anchor_coverage(
        keyframes=(keyframe("b"), keyframe("a")),
        exported_actor_identities=(),
    )

    assert result.source_keyframe_count == 2
    assert result.anchor_with_actor_rows_count == 0
    assert result.anchor_without_actor_rows == ("a", "b")


def test_duplicate_source_anchor_is_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        summarize_actor_export_anchor_coverage(
            keyframes=(keyframe("a"), keyframe("a")),
            exported_actor_identities=(),
        )


def test_duplicate_export_identity_is_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        summarize_actor_export_anchor_coverage(
            keyframes=(keyframe("a"),),
            exported_actor_identities=(("a", "1"), ("a", "1")),
        )


def test_exported_anchor_absent_from_source_is_rejected():
    with pytest.raises(ValueError, match="absent from source"):
        summarize_actor_export_anchor_coverage(
            keyframes=(keyframe("a"),),
            exported_actor_identities=(("b", "1"),),
        )


def test_missing_source_anchor_id_is_rejected():
    with pytest.raises(ValueError, match="missing anchor_id"):
        summarize_actor_export_anchor_coverage(
            keyframes=({},),
            exported_actor_identities=(),
        )
