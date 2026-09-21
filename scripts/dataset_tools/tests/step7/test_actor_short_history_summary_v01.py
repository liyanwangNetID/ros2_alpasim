import pytest

from step7.actor_short_history_summary_v01 import summarize_actor_short_history_rows


def test_summary_preserves_rowless_keyframes_and_statuses():
    result = summarize_actor_short_history_rows(
        keyframes=({"anchor_id": "a"}, {"anchor_id": "b"}),
        rows=({"anchor_id": "a", "track_id": "1", "history_status": "usable", "label_class": "automobile"},),
    )
    assert result["keyframe_count"] == 2
    assert result["actor_row_count"] == 1
    assert result["anchor_without_actor_rows_count"] == 1
    assert result["history_status_counts"] == {"usable": 1}


def test_duplicate_identity_is_rejected():
    row = {"anchor_id": "a", "track_id": "1", "history_status": "usable", "label_class": "automobile"}
    with pytest.raises(ValueError, match="identities must be unique"):
        summarize_actor_short_history_rows(keyframes=({"anchor_id": "a"},), rows=(row, dict(row)))
