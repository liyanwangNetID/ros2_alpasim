import pytest

from step7.actor_observability_selected_policy_impact_v01 import (
    analyze_selected_policy_changes,
)


def row(track, *, winning, fraction, actor_class="automobile", clip="c"):
    return {
        "anchor_id": "a",
        "clip_id": clip,
        "track_id": track,
        "label_class": actor_class,
        "evidence_status": "combined_evidence_available",
        "total_winning_cell_count": winning,
        "maximum_visible_fraction": fraction,
        "static_occlusion_evaluated": False,
    }


def test_classifies_candidate_disagreements_and_summarizes_clips():
    report = analyze_selected_policy_changes(
        evidence_rows=(
            row("1", winning=1, fraction=0.5),
            row("2", winning=5, fraction=0.005),
            row("3", winning=1, fraction=0.005, actor_class="person", clip="d"),
            row("4", winning=5, fraction=0.5),
        )
    )
    assert report["affected_actor_count"] == 3
    assert report["affected_clip_count"] == 2
    assert report["change_category_counts"] == {
        "rejected_by_both_candidates": 1,
        "rejected_by_visible_fraction_candidate_only": 1,
        "rejected_by_winning_cell_candidate_only": 1,
    }
    assert report["affected_by_winning_cell_candidate_count"] == 2
    assert report["affected_by_visible_fraction_candidate_count"] == 2
    assert report["by_clip"]["c"]["affected_actor_count"] == 2
    assert tuple(item["track_id"] for item in report["affected_rows"]) == ("1", "2", "3")


def test_expected_production_count_mismatch_is_rejected():
    with pytest.raises(RuntimeError, match="winning-cell candidate affected count changed"):
        analyze_selected_policy_changes(
            evidence_rows=(row("1", winning=1, fraction=0.5),),
            expected_winning_affected_count=361,
        )


def test_missing_clip_id_is_rejected():
    value = row("1", winning=1, fraction=0.5)
    del value["clip_id"]
    with pytest.raises(ValueError, match="missing clip_id"):
        analyze_selected_policy_changes(evidence_rows=(value,))


def test_duplicate_identity_is_rejected():
    value = row("1", winning=1, fraction=0.5)
    with pytest.raises(ValueError, match="identities must be unique"):
        analyze_selected_policy_changes(evidence_rows=(value, dict(value)))
