from types import SimpleNamespace

import pytest

from step7.actor_geometric_occlusion_export_inputs_v02 import (
    prepare_actor_geometric_occlusion_projection_context_export_inputs,
)


def keyframe():
    return {
        "anchor_id": "clip_100",
        "clip_id": "clip",
        "anchor_ns": 100,
    }


def context(track_id, actor_class="automobile"):
    return SimpleNamespace(
        track_id=track_id,
        combined_evidence=SimpleNamespace(actor_class=actor_class),
    )


def result(*track_ids):
    values = tuple(context(track_id) for track_id in track_ids)
    return SimpleNamespace(
        actor_count=len(values),
        projection_context=SimpleNamespace(
            actor_count=len(values),
            actor_context=values,
        ),
    )


def test_prepares_v02_inputs_in_track_order():
    value = prepare_actor_geometric_occlusion_projection_context_export_inputs(
        keyframe=keyframe(),
        actors=(
            {"track_id": "2", "label_class": "automobile", "is_static": True},
            {"track_id": "1", "label_class": "automobile", "is_static": False},
        ),
        result=result("1", "2"),
    )

    assert tuple(item.context.track_id for item in value) == ("1", "2")
    assert tuple(item.is_static for item in value) == (False, True)
    assert all(item.keyframe is value[0].keyframe for item in value)


def test_actor_set_mismatch_is_rejected():
    with pytest.raises(ValueError, match="sets must match"):
        prepare_actor_geometric_occlusion_projection_context_export_inputs(
            keyframe=keyframe(),
            actors=(
                {"track_id": "1", "label_class": "automobile", "is_static": False},
            ),
            result=result("2"),
        )


def test_duplicate_actor_track_id_is_rejected():
    actor = {"track_id": "1", "label_class": "automobile", "is_static": False}
    with pytest.raises(ValueError, match="duplicate track_id"):
        prepare_actor_geometric_occlusion_projection_context_export_inputs(
            keyframe=keyframe(),
            actors=(actor, actor),
            result=result("1"),
        )


def test_missing_is_static_is_rejected():
    with pytest.raises(ValueError, match="missing is_static"):
        prepare_actor_geometric_occlusion_projection_context_export_inputs(
            keyframe=keyframe(),
            actors=({"track_id": "1", "label_class": "automobile"},),
            result=result("1"),
        )


def test_actor_class_mismatch_is_rejected():
    with pytest.raises(ValueError, match="Actor class"):
        prepare_actor_geometric_occlusion_projection_context_export_inputs(
            keyframe=keyframe(),
            actors=(
                {"track_id": "1", "label_class": "person", "is_static": False},
            ),
            result=result("1"),
        )


def test_result_count_mismatch_is_rejected():
    value = result("1")
    value.actor_count = 2
    with pytest.raises(ValueError, match="count are inconsistent"):
        prepare_actor_geometric_occlusion_projection_context_export_inputs(
            keyframe=keyframe(),
            actors=(
                {"track_id": "1", "label_class": "automobile", "is_static": False},
            ),
            result=value,
        )


def test_empty_anchor_is_supported():
    value = prepare_actor_geometric_occlusion_projection_context_export_inputs(
        keyframe=keyframe(),
        actors=(),
        result=result(),
    )

    assert value == ()
