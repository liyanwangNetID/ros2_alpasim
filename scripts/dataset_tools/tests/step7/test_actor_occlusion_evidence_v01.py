import pytest

from step7.actor_occlusion_evidence_v01 import (
    OCCLUSION_EVIDENCE_FORMAT_VERSION,
    build_camera_actor_occlusion_evidence,
)


def build(**overrides):
    values = {
        "camera_name": "cross_left",
        "track_id": "226",
        "raster_width": 480,
        "raster_height": 270,
        "occupied_cell_count": 60,
        "winning_cell_count": 0,
        "occluded_cell_count": 60,
        "occluding_actor_ids": ("80", "17", "132", "18"),
        "actor_to_actor_occlusion_evaluated": True,
        "static_occlusion_evaluated": False,
        "reasons": (),
    }
    values.update(overrides)
    return build_camera_actor_occlusion_evidence(**values)


def test_evaluated_evidence_calculates_fraction():
    result = build(
        occupied_cell_count=100,
        winning_cell_count=25,
        occluded_cell_count=75,
    )
    assert result.visible_fraction == 0.25
    assert result.evidence_status == "evaluated"
    assert result.actor_to_actor_occlusion_evaluated
    assert not result.static_occlusion_evaluated


def test_fully_occluded_evidence_preserves_zero():
    result = build()
    assert result.visible_fraction == 0.0
    assert result.evidence_status == "evaluated"


def test_no_sampled_surface_has_undefined_fraction():
    result = build(
        occupied_cell_count=0,
        winning_cell_count=0,
        occluded_cell_count=0,
        occluding_actor_ids=(),
    )
    assert result.visible_fraction is None
    assert result.evidence_status == "no_sampled_surface"


def test_unevaluated_evidence_has_explicit_status():
    result = build(
        occupied_cell_count=0,
        winning_cell_count=0,
        occluded_cell_count=0,
        occluding_actor_ids=(),
        actor_to_actor_occlusion_evaluated=False,
        reasons=("camera_input_unavailable",),
    )
    assert result.visible_fraction is None
    assert result.evidence_status == "not_evaluated"


def test_occluding_actor_ids_are_sorted():
    result = build()
    assert result.occluding_actor_ids == ("132", "17", "18", "80")


def test_to_dict_uses_json_ready_lists():
    result = build()
    value = result.to_dict()
    assert value["occlusion_evidence_format_version"] == OCCLUSION_EVIDENCE_FORMAT_VERSION
    assert isinstance(value["occluding_actor_ids"], list)
    assert isinstance(value["reasons"], list)


def test_inconsistent_counts_are_rejected():
    with pytest.raises(ValueError, match="must equal occupied"):
        build(
            occupied_cell_count=60,
            winning_cell_count=1,
            occluded_cell_count=60,
        )


def test_negative_counts_are_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        build(occupied_cell_count=-1)


def test_unevaluated_counts_are_rejected():
    with pytest.raises(ValueError, match="cannot contain Z-buffer counts"):
        build(actor_to_actor_occlusion_evaluated=False)


def test_self_occlusion_is_rejected():
    with pytest.raises(ValueError, match="cannot occlude itself"):
        build(occluding_actor_ids=("226",))


def test_duplicate_occluders_are_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        build(occluding_actor_ids=("17", "17"))


@pytest.mark.parametrize(
    "field,value",
    [
        ("raster_width", 0),
        ("raster_height", -1),
    ],
)
def test_invalid_raster_dimensions_are_rejected(field, value):
    with pytest.raises(ValueError, match="must be positive"):
        build(**{field: value})
