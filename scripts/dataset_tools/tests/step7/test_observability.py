"""Tests for the consolidated Step 7 observability domain."""
from __future__ import annotations
import math
import pytest
from step7.observability import FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO, FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT, FAILURE_BELOW_PRIMARY_AREA_AND_HEIGHT, evaluate_geometric_observability
from dataclasses import replace
from step7.projection import ActorCameraProjection
from step7.observability import aggregate_actor_observability
from step7.scene_facts import CAMERA_NAMES, OBSERVABILITY_FORMAT_VERSION
from types import SimpleNamespace
from step7.scene_facts import CAMERA_NAMES
from step7.observability import FROZEN_DYNAMIC_OCCLUSION_POLICY, MINIMUM_VISIBLE_FRACTION, MINIMUM_WINNING_CELL_COUNT, POLICY_NAME, evaluate_frozen_dynamic_occlusion_policy
from step7.observability import ObservabilityShadowPolicy, evaluate_actor_observability_shadow, evaluate_actor_observability_shadow_policies
from step7 import observability as target
from step7.observability import FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO, FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT, evaluate_geometric_observability

def rules_evaluate(camera_name: str, *, area: float=100.0, height: float=10.0, ratio: float=1.0):
    return evaluate_geometric_observability(camera_name=camera_name, inside_image_hull_area_px=area, projected_height_px=height, inside_image_hull_ratio=ratio)

@pytest.mark.parametrize(('camera_name', 'threshold'), (('front_wide', 5.0), ('cross_left', 6.0), ('cross_right', 6.0)))
def test_height_policy_boundary_is_inclusive(camera_name, threshold):
    decision = rules_evaluate(camera_name, height=threshold)
    assert decision.candidate is True
    assert decision.failure_reason is None

@pytest.mark.parametrize(('camera_name', 'threshold'), (('front_wide', 5.0), ('cross_left', 6.0), ('cross_right', 6.0)))
def test_height_policy_rejects_value_immediately_below_boundary(camera_name, threshold):
    decision = rules_evaluate(camera_name, height=math.nextafter(threshold, 0.0))
    assert decision.candidate is False
    assert decision.failure_reason == FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT

def test_wide_and_cross_height_policy_does_not_require_area_or_ratio():
    decision = rules_evaluate('front_wide', area=0.0, height=5.0, ratio=0.0)
    assert decision.candidate is True
    assert decision.failure_reason is None

def test_front_tele_area_branch_passes_at_boundary():
    decision = rules_evaluate('front_tele', area=512.0, height=0.0, ratio=0.1)
    assert decision.candidate is True
    assert decision.failure_reason is None

def test_front_tele_height_branch_preserves_complete_slender_target():
    decision = rules_evaluate('front_tele', area=126.3, height=17.06, ratio=1.0)
    assert decision.candidate is True
    assert decision.failure_reason is None

def test_front_tele_rejects_when_both_scale_features_are_below_threshold():
    decision = rules_evaluate('front_tele', area=math.nextafter(512.0, 0.0), height=math.nextafter(16.0, 0.0), ratio=1.0)
    assert decision.candidate is False
    assert decision.failure_reason == FAILURE_BELOW_PRIMARY_AREA_AND_HEIGHT

def test_front_tele_ratio_boundary_is_inclusive():
    decision = rules_evaluate('front_tele', area=512.0, height=16.0, ratio=0.1)
    assert decision.candidate is True
    assert decision.failure_reason is None

def test_front_tele_rejects_scale_pass_with_ratio_below_boundary():
    decision = rules_evaluate('front_tele', area=1000.0, height=100.0, ratio=math.nextafter(0.1, 0.0))
    assert decision.candidate is False
    assert decision.failure_reason == FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO

def test_front_tele_does_not_allow_large_area_to_bypass_ratio_guard():
    decision = rules_evaluate('front_tele', area=10000.0, height=200.0, ratio=0.01)
    assert decision.candidate is False
    assert decision.failure_reason == FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO

@pytest.mark.parametrize('camera_name', ('rear', '', 'FRONT_WIDE'))
def test_unknown_camera_is_rejected(camera_name):
    with pytest.raises(ValueError, match='Unsupported camera_name'):
        rules_evaluate(camera_name)

@pytest.mark.parametrize(('field', 'value', 'message'), (('area', -1.0, 'inside_image_hull_area_px'), ('height', -1.0, 'projected_height_px'), ('ratio', -0.01, 'inside_image_hull_ratio'), ('ratio', 1.01, 'inside_image_hull_ratio')))
def test_invalid_numeric_range_is_rejected(field, value, message):
    arguments = {'area': 100.0, 'height': 10.0, 'ratio': 1.0}
    arguments[field] = value
    with pytest.raises(ValueError, match=message):
        rules_evaluate('front_wide', **arguments)

def aggregate_projection(camera_name: str, *, track_id: str='18', actor_class: str='automobile', valid: bool=True, failure_reason: str | None=None, area: float=1000.0, height: float=20.0, ratio: float=1.0) -> ActorCameraProjection:
    return ActorCameraProjection(camera_name=camera_name, track_id=track_id, actor_class=actor_class, corner_count=8, edge_count=12, edge_samples_per_edge=0, camera_sample_count=8, positive_depth_sample_count=8 if valid else 0, within_fov_sample_count=8 if valid else 0, inside_image_sample_count=8 if valid else 0, projected_bbox=None, clipped_bbox=None, projected_area_px=area if valid else 0.0, inside_image_area_px=area if valid else 0.0, inside_image_ratio=ratio if valid else 0.0, projected_hull=(), clipped_hull=(), projected_hull_area_px=area if valid else 0.0, inside_image_hull_area_px=area if valid else 0.0, inside_image_hull_ratio=ratio if valid else 0.0, projected_height_px=height if valid else 0.0, minimum_depth_m=10.0 if valid else None, maximum_depth_m=12.0 if valid else None, truncated=valid and ratio < 1.0, projection_valid=valid, failure_reason=failure_reason)

def aggregate_all_projections(**overrides):
    return {name: overrides.get(name, aggregate_projection(name)) for name in CAMERA_NAMES}

def test_aggregates_visible_cameras_in_canonical_order():
    values = aggregate_all_projections(front_wide=aggregate_projection('front_wide', height=4.9), cross_left=aggregate_projection('cross_left', height=6.0))
    result = aggregate_actor_observability(values)
    assert result.observability_status == 'candidate_visible'
    assert result.visible_in_cameras == ('front_tele', 'cross_left', 'cross_right')
    assert result.actor_to_actor_occlusion_evaluated is False
    assert result.static_occlusion_evaluated is False

def test_all_conclusive_camera_failures_produce_not_visible():
    values = aggregate_all_projections(**{name: aggregate_projection(name, valid=False, failure_reason='box_outside_camera_fov') for name in CAMERA_NAMES})
    result = aggregate_actor_observability(values)
    assert result.observability_status == 'not_visible'
    assert result.visible_in_cameras == ()
    assert all((not item.geometric_observability_candidate for item in result.camera_observability))

def test_valid_but_small_projections_produce_not_visible():
    values = aggregate_all_projections(front_wide=aggregate_projection('front_wide', height=4.9), front_tele=aggregate_projection('front_tele', area=511.0, height=15.9), cross_left=aggregate_projection('cross_left', height=5.9), cross_right=aggregate_projection('cross_right', height=5.9))
    result = aggregate_actor_observability(values)
    assert result.observability_status == 'not_visible'
    assert result.visible_in_cameras == ()
    assert all((item.failure_reason for item in result.camera_observability))

def test_to_dict_uses_json_ready_lists_and_version():
    result = aggregate_actor_observability(aggregate_all_projections())
    value = result.to_dict()
    assert value['observability_format_version'] == OBSERVABILITY_FORMAT_VERSION
    assert value['visible_in_cameras'] == list(CAMERA_NAMES)
    assert isinstance(value['camera_observability'], list)
    assert len(value['camera_observability']) == 4

def test_missing_camera_is_rejected():
    values = aggregate_all_projections()
    del values['cross_right']
    with pytest.raises(ValueError, match='Expected exactly CAMERA_NAMES'):
        aggregate_actor_observability(values)

def test_extra_camera_is_rejected():
    values = aggregate_all_projections()
    values['rear'] = aggregate_projection('rear')
    with pytest.raises(ValueError, match='Expected exactly CAMERA_NAMES'):
        aggregate_actor_observability(values)

def test_mismatched_projection_camera_is_rejected():
    values = aggregate_all_projections()
    values['front_wide'] = aggregate_projection('cross_left')
    with pytest.raises(ValueError, match='Projection camera mismatch'):
        aggregate_actor_observability(values)

def test_mixed_track_ids_are_rejected():
    values = aggregate_all_projections()
    values['front_wide'] = aggregate_projection('front_wide', track_id='other')
    with pytest.raises(ValueError, match='same track_id'):
        aggregate_actor_observability(values)

def test_mixed_actor_classes_are_rejected():
    values = aggregate_all_projections()
    values['front_wide'] = aggregate_projection('front_wide', actor_class='person')
    with pytest.raises(ValueError, match='same actor_class'):
        aggregate_actor_observability(values)

def test_invalid_projection_requires_failure_reason():
    values = aggregate_all_projections()
    values['front_wide'] = aggregate_projection('front_wide', valid=False)
    with pytest.raises(ValueError, match='must provide a failure_reason'):
        aggregate_actor_observability(values)

def test_valid_projection_forbids_failure_reason():
    values = aggregate_all_projections()
    values['front_wide'] = aggregate_projection('front_wide', valid=True, failure_reason='unexpected')
    with pytest.raises(ValueError, match='must not provide a failure_reason'):
        aggregate_actor_observability(values)

def multicamera_actor(track_id, actor_class='automobile'):
    return {'track_id': track_id, 'label_class': actor_class}

def multicamera_projection(camera_name, track_id, actor_class, valid=True):
    return ActorCameraProjection(camera_name=camera_name, track_id=track_id, actor_class=actor_class, corner_count=8, edge_count=12, edge_samples_per_edge=0, camera_sample_count=1 if valid else 0, positive_depth_sample_count=1 if valid else 0, within_fov_sample_count=1 if valid else 0, inside_image_sample_count=1 if valid else 0, projected_bbox=None, clipped_bbox=None, projected_area_px=0.0, inside_image_area_px=0.0, inside_image_ratio=0.0, projected_hull=(), clipped_hull=(), projected_hull_area_px=0.0, inside_image_hull_area_px=600.0 if valid else 0.0, inside_image_hull_ratio=1.0 if valid else 0.0, projected_height_px=20.0 if valid else 0.0, minimum_depth_m=10.0 if valid else None, maximum_depth_m=12.0 if valid else None, truncated=False, projection_valid=valid, failure_reason=None if valid else 'box_outside_camera_fov')

def multicamera_calibrations():
    return {name: SimpleNamespace(camera_name=name) for name in CAMERA_NAMES}

def multicamera_install_projection_stub(monkeypatch, invalid_camera=None):
    calls = []

    def project(value, *, calibration, **kwargs):
        calls.append((value['track_id'], calibration.camera_name, kwargs))
        return multicamera_projection(calibration.camera_name, str(value['track_id']), str(value['label_class']), valid=calibration.camera_name != invalid_camera)
    monkeypatch.setattr(target, 'project_actor_box_to_camera', project)
    return calls

def multicamera_call(monkeypatch, **overrides):
    calls = multicamera_install_projection_stub(monkeypatch)
    values = {'actors': (multicamera_actor('2'), multicamera_actor('1', 'person')), 'recorded_ego_message': {'ego': True}, 'calibrations': multicamera_calibrations()}
    values.update(overrides)
    result = target.build_multicamera_actor_geometric_observability(**values)
    return (result, calls)

def test_builds_all_actor_projections_and_observability(monkeypatch):
    result, calls = multicamera_call(monkeypatch)
    assert result.actor_count == 2
    assert tuple((item.track_id for item in result.actor_results)) == ('1', '2')
    assert tuple((item.track_id for item in result.actor_observability)) == ('1', '2')
    assert len(calls) == 2 * len(CAMERA_NAMES)
    assert all((item.observability.observability_status == 'candidate_visible' for item in result.actor_results))

def test_projection_and_camera_order_is_canonical(monkeypatch):
    result, _ = multicamera_call(monkeypatch)
    for item in result.actor_results:
        assert tuple((multicamera_projection.camera_name for multicamera_projection in item.projections)) == tuple(CAMERA_NAMES)
        assert item.observability.visible_in_cameras == tuple(CAMERA_NAMES)

def test_projection_parameters_are_forwarded(monkeypatch):
    result, calls = multicamera_call(monkeypatch, samples_per_edge=9, near_plane_m=0.01, maximum_chord_error_px=0.5, maximum_adaptive_depth=11)
    assert result.actor_count == 2
    for _, _, kwargs in calls:
        assert kwargs['samples_per_edge'] == 9
        assert kwargs['near_plane_m'] == 0.01
        assert kwargs['maximum_chord_error_px'] == 0.5
        assert kwargs['maximum_adaptive_depth'] == 11

def test_invalid_projection_is_aggregated_not_dropped(monkeypatch):
    invalid_camera = CAMERA_NAMES[-1]
    multicamera_install_projection_stub(monkeypatch, invalid_camera=invalid_camera)
    result = target.build_multicamera_actor_geometric_observability(actors=(multicamera_actor('1'),), recorded_ego_message={'ego': True}, calibrations=multicamera_calibrations())
    assert result.actor_count == 1
    assert invalid_camera not in result.actor_observability[0].visible_in_cameras
    assert result.actor_observability[0].observability_status == 'candidate_visible'

def test_multicamera_missing_camera_is_rejected(monkeypatch):
    source = multicamera_calibrations()
    del source[CAMERA_NAMES[0]]
    with pytest.raises(ValueError, match='missing'):
        multicamera_call(monkeypatch, calibrations=source)

def test_multicamera_extra_camera_is_rejected(monkeypatch):
    source = multicamera_calibrations()
    source['extra'] = SimpleNamespace(camera_name='extra')
    with pytest.raises(ValueError, match='extra'):
        multicamera_call(monkeypatch, calibrations=source)

def test_duplicate_track_ids_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match='must be unique'):
        multicamera_call(monkeypatch, actors=(multicamera_actor('1'), multicamera_actor('1')))

def test_missing_track_id_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match='usable track_id'):
        multicamera_call(monkeypatch, actors=({'label_class': 'automobile'},))

def test_missing_actor_class_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match='usable label_class'):
        multicamera_call(monkeypatch, actors=({'track_id': '1'},))

def test_empty_actor_set_is_supported(monkeypatch):
    result, calls = multicamera_call(monkeypatch, actors=())
    assert result.actor_count == 0
    assert result.actor_results == ()
    assert result.actor_observability == ()
    assert calls == []

def frozen_row(track_id, winning, visible_fraction):
    return {'anchor_id': 'a', 'clip_id': 'c', 'track_id': track_id, 'label_class': 'automobile', 'evidence_status': 'combined_evidence_available', 'total_winning_cell_count': winning, 'maximum_visible_fraction': visible_fraction, 'static_occlusion_evaluated': False}

def test_frozen_policy_constants_are_explicit():
    assert POLICY_NAME == 'step7e_dynamic_occlusion_v01'
    assert MINIMUM_WINNING_CELL_COUNT == 2
    assert MINIMUM_VISIBLE_FRACTION == 0.0
    assert FROZEN_DYNAMIC_OCCLUSION_POLICY.minimum_winning_cell_count == 2
    assert FROZEN_DYNAMIC_OCCLUSION_POLICY.minimum_visible_fraction == 0.0

def test_two_winning_cells_are_visible_even_with_low_fraction():
    decisions = evaluate_frozen_dynamic_occlusion_policy(evidence_rows=(frozen_row('1', 2, 0.0001),))
    assert len(decisions) == 1
    assert decisions[0].policy_name == POLICY_NAME
    assert decisions[0].shadow_status == 'shadow_visible'

def test_one_winning_cell_is_not_visible_even_with_high_fraction():
    decisions = evaluate_frozen_dynamic_occlusion_policy(evidence_rows=(frozen_row('1', 1, 1.0),))
    assert decisions[0].shadow_status == 'shadow_not_visible'

def shadow_policy(winning=1, fraction=0.1, name='p1'):
    return ObservabilityShadowPolicy(policy_name=name, minimum_winning_cell_count=winning, minimum_visible_fraction=fraction)

def shadow_evidence(*, status='combined_evidence_available', winning=5, fraction=0.5, anchor='a', track='1'):
    return {'anchor_id': anchor, 'track_id': track, 'evidence_status': status, 'total_winning_cell_count': winning, 'maximum_visible_fraction': fraction, 'static_occlusion_evaluated': False}

def test_combined_evidence_is_shadow_visible_when_both_requirements_pass():
    result = evaluate_actor_observability_shadow(evidence=shadow_evidence(), policy=shadow_policy())
    assert result.shadow_status == 'shadow_visible'
    assert result.winning_cell_requirement_met
    assert result.visible_fraction_requirement_met
    assert result.reasons == ('dynamic_occlusion_policy_requirements_met',)

def test_combined_evidence_reports_each_failed_requirement():
    result = evaluate_actor_observability_shadow(evidence=shadow_evidence(winning=0, fraction=0.05), policy=shadow_policy())
    assert result.shadow_status == 'shadow_not_visible'
    assert result.reasons == ('minimum_winning_cell_count_not_met', 'minimum_visible_fraction_not_met')

def test_no_geometric_candidate_is_shadow_not_visible():
    result = evaluate_actor_observability_shadow(evidence=shadow_evidence(status='no_geometric_candidate', winning=10, fraction=1.0), policy=shadow_policy())
    assert result.shadow_status == 'shadow_not_visible'
    assert result.reasons == ('no_geometric_candidate',)

def test_candidate_without_surface_is_shadow_indeterminate():
    result = evaluate_actor_observability_shadow(evidence=shadow_evidence(status='candidate_without_sampled_surface', winning=0, fraction=None), policy=shadow_policy())
    assert result.shadow_status == 'shadow_indeterminate'
    assert result.reasons == ('geometric_candidate_without_sampled_surface',)

def test_threshold_boundaries_are_inclusive():
    result = evaluate_actor_observability_shadow(evidence=shadow_evidence(winning=2, fraction=0.25), policy=shadow_policy(winning=2, fraction=0.25))
    assert result.shadow_status == 'shadow_visible'

def test_policy_grid_is_deterministic():
    result = evaluate_actor_observability_shadow_policies(evidence_rows=(shadow_evidence(anchor='b', track='2'), shadow_evidence(anchor='a', track='9')), policies=(shadow_policy(name='first'), shadow_policy(name='second')))
    assert tuple(((item.policy_name, item.anchor_id, item.track_id) for item in result)) == (('first', 'a', '9'), ('first', 'b', '2'), ('second', 'a', '9'), ('second', 'b', '2'))

def test_duplicate_policy_name_is_rejected():
    with pytest.raises(ValueError, match='policy names must be unique'):
        evaluate_actor_observability_shadow_policies(evidence_rows=(shadow_evidence(),), policies=(shadow_policy(), shadow_policy()))

def test_static_occlusion_result_is_rejected():
    row = shadow_evidence()
    row['static_occlusion_evaluated'] = True
    with pytest.raises(ValueError, match='static occlusion'):
        evaluate_actor_observability_shadow(evidence=row, policy=shadow_policy())

def test_invalid_policy_threshold_is_rejected():
    with pytest.raises(ValueError, match='within'):
        shadow_policy(fraction=1.1)

def projection_adapter_make_projection(*, camera_name: str, area: float, height: float, ratio: float) -> ActorCameraProjection:
    """Construct a minimal valid-shaped projection for rule adaptation tests."""
    return ActorCameraProjection(camera_name=camera_name, track_id='test-track', actor_class='automobile', corner_count=8, edge_count=12, edge_samples_per_edge=0, camera_sample_count=8, positive_depth_sample_count=8, within_fov_sample_count=8, inside_image_sample_count=8, projected_bbox=None, clipped_bbox=None, projected_area_px=area, inside_image_area_px=area, inside_image_ratio=ratio, projected_hull=(), clipped_hull=(), projected_hull_area_px=area, inside_image_hull_area_px=area, inside_image_hull_ratio=ratio, projected_height_px=height, minimum_depth_m=10.0, maximum_depth_m=12.0, truncated=ratio < 1.0, projection_valid=True, failure_reason=None)

def projection_adapter_evaluate_projection(projection: ActorCameraProjection):
    """Pass production projection fields directly into the pure rule."""
    assert projection.projection_valid is True
    return evaluate_geometric_observability(camera_name=projection.camera_name, inside_image_hull_area_px=projection.inside_image_hull_area_px, projected_height_px=projection.projected_height_px, inside_image_hull_ratio=projection.inside_image_hull_ratio)

@pytest.mark.parametrize(('camera_name', 'height'), (('front_wide', 5.0), ('cross_left', 6.0), ('cross_right', 6.0)))
def test_projection_fields_directly_drive_height_policies(camera_name, height):
    projection = projection_adapter_make_projection(camera_name=camera_name, area=0.0, height=height, ratio=0.0)
    decision = projection_adapter_evaluate_projection(projection)
    assert decision.candidate is True
    assert decision.failure_reason is None

def test_projection_fields_directly_drive_height_rejection():
    projection = projection_adapter_make_projection(camera_name='cross_left', area=10000.0, height=5.99, ratio=1.0)
    decision = projection_adapter_evaluate_projection(projection)
    assert decision.candidate is False
    assert decision.failure_reason == FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT

def test_projection_fields_directly_drive_front_tele_area_branch():
    projection = projection_adapter_make_projection(camera_name='front_tele', area=512.0, height=10.0, ratio=0.1)
    decision = projection_adapter_evaluate_projection(projection)
    assert decision.candidate is True
    assert decision.failure_reason is None

def test_projection_fields_directly_drive_front_tele_slender_height_branch():
    projection = projection_adapter_make_projection(camera_name='front_tele', area=126.3, height=17.06, ratio=1.0)
    decision = projection_adapter_evaluate_projection(projection)
    assert decision.candidate is True
    assert decision.failure_reason is None

def test_projection_fields_directly_drive_front_tele_fragment_rejection():
    projection = projection_adapter_make_projection(camera_name='front_tele', area=10000.0, height=200.0, ratio=0.099)
    decision = projection_adapter_evaluate_projection(projection)
    assert decision.candidate is False
    assert decision.failure_reason == FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO

def test_projection_to_dict_preserves_observability_field_names_and_values():
    projection = projection_adapter_make_projection(camera_name='front_tele', area=512.0, height=16.0, ratio=0.1)
    value = projection.to_dict()
    assert value['camera_name'] == 'front_tele'
    assert value['inside_image_hull_area_px'] == 512.0
    assert value['projected_height_px'] == 16.0
    assert value['inside_image_hull_ratio'] == 0.1
