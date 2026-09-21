import pytest

from step7.actor_observability_policy_visual_review_v01 import (
    prepare_policy_visual_review_cases,
)


def selected():
    return {
        "schema_version": "step7e-observability-policy-review-selection-v01",
        "cases": [{
            "anchor_id": "a", "track_id": "1", "clip_id": "c",
            "label_class": "automobile", "review_category": "both",
        }],
    }


def v02(cameras=("front_wide",)):
    return {
        "schema_version": "step7e-geometric-occlusion-evidence-v02",
        "anchor_id": "a", "track_id": "1", "clip_id": "c",
        "anchor_ns": 10, "label_class": "automobile",
        "occlusion_winning_camera_names": list(cameras),
        "occluding_actor_ids": ["2"],
        "total_occupied_cell_count": 5,
        "total_occluded_cell_count": 4,
    }


def projection():
    return {
        "anchor_id": "a", "track_id": "1", "camera_name": "front_wide",
        "anchor_ns": 10, "label_class": "automobile",
        "clipped_bbox": {"min_u": 1, "min_v": 2, "max_u": 3, "max_v": 4},
        "clipped_hull": [{"u": 1, "v": 2}, {"u": 3, "v": 4}],
        "inside_image_hull_area_px": 4,
        "inside_image_hull_ratio": 0.5,
        "projected_height_px": 2,
        "minimum_depth_m": 8,
    }


def test_joins_single_winning_camera_and_projection():
    report = prepare_policy_visual_review_cases(
        selection=selected(), v02_rows=(v02(),), projection_rows=(projection(),)
    )
    assert report["prepared_case_count"] == 1
    assert report["failure_count"] == 0
    case = report["cases"][0]
    assert case["camera_name"] == "front_wide"
    assert case["camera_selection_basis"] == "single_occlusion_winning_camera"
    assert case["anchor_ns"] == 10
    assert case["occluding_actor_ids"] == ["2"]


def test_reports_nonunique_winning_camera_without_guessing():
    report = prepare_policy_visual_review_cases(
        selection=selected(),
        v02_rows=(v02(("front_wide", "cross_left")),),
        projection_rows=(projection(),),
    )
    assert report["prepared_case_count"] == 0
    assert report["failure_counts"] == {"winning_camera_count_not_one": 1}


def test_reports_missing_projection_and_class_mismatch():
    report = prepare_policy_visual_review_cases(
        selection=selected(), v02_rows=(v02(),), projection_rows=()
    )
    assert report["failure_counts"] == {"missing_projection_row": 1}
    changed = v02()
    changed["label_class"] = "person"
    report = prepare_policy_visual_review_cases(
        selection=selected(), v02_rows=(changed,), projection_rows=(projection(),)
    )
    assert report["failure_counts"] == {"actor_class_mismatch": 1}


def test_rejects_duplicate_selected_identity():
    value = selected()
    value["cases"] = value["cases"] * 2
    with pytest.raises(ValueError, match="must be unique"):
        prepare_policy_visual_review_cases(
            selection=value, v02_rows=(v02(),), projection_rows=(projection(),)
        )
