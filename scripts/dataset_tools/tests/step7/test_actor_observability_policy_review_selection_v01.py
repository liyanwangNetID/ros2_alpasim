import pytest

from step7.actor_observability_policy_review_selection_v01 import (
    BOTH,
    FRACTION_ONLY,
    WINNING_ONLY,
    select_policy_review_cases,
)


def row(anchor, track, clip, category, winning, fraction):
    return {
        "anchor_id": anchor,
        "clip_id": clip,
        "track_id": track,
        "label_class": "automobile",
        "maximum_visible_fraction": fraction,
        "total_winning_cell_count": winning,
        "change_category": category,
    }


def report(rows):
    return {
        "schema_version": "step7e-observability-selected-policy-impact-v01",
        "affected_rows": rows,
    }


def test_selects_balanced_clip_diverse_cases_deterministically():
    rows = [
        row("a1", "1", "c1", WINNING_ONLY, 1, 1.0),
        row("a2", "2", "c1", WINNING_ONLY, 1, 0.5),
        row("a3", "3", "c2", WINNING_ONLY, 1, 0.2),
        row("a4", "4", "c3", FRACTION_ONLY, 20, 0.009),
        row("a5", "5", "c4", FRACTION_ONLY, 10, 0.001),
        row("a6", "6", "c5", BOTH, 1, 0.009),
        row("a7", "7", "c6", BOTH, 1, 0.001),
    ]
    result = select_policy_review_cases(
        impact_report=report(rows), quota_per_category=2
    )
    assert result["selected_case_count"] == 6
    assert result["selected_clip_count"] == 6
    assert result["selected_category_counts"] == {
        WINNING_ONLY: 2,
        FRACTION_ONLY: 2,
        BOTH: 2,
    }
    winning = [
        case for case in result["cases"] if case["review_category"] == WINNING_ONLY
    ]
    assert [(case["anchor_id"], case["track_id"]) for case in winning] == [
        ("a1", "1"),
        ("a3", "3"),
    ]
    assert result == select_policy_review_cases(
        impact_report=report(tuple(reversed(rows))), quota_per_category=2
    )


def test_fills_quota_when_unique_clips_are_exhausted():
    rows = [
        row("a1", "1", "c1", WINNING_ONLY, 1, 0.8),
        row("a2", "2", "c1", WINNING_ONLY, 1, 0.7),
    ]
    result = select_policy_review_cases(
        impact_report=report(rows), quota_per_category=2
    )
    assert result["selected_case_count"] == 2
    assert result["selected_clip_count"] == 1


def test_rejects_duplicate_identity():
    value = row("a", "1", "c", BOTH, 1, 0.001)
    with pytest.raises(ValueError, match="identities must be unique"):
        select_policy_review_cases(
            impact_report=report((value, dict(value))), quota_per_category=1
        )


def test_rejects_nonpositive_quota_and_wrong_schema():
    with pytest.raises(ValueError, match="must be positive"):
        select_policy_review_cases(impact_report=report(()), quota_per_category=0)
    with pytest.raises(ValueError, match="unexpected impact report schema"):
        select_policy_review_cases(
            impact_report={"schema_version": "wrong", "affected_rows": []}
        )
