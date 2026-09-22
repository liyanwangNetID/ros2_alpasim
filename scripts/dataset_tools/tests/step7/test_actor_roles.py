"""Tests for the consolidated Step 7 Actor-role selection domain."""
from __future__ import annotations
from step7.actor_roles import ActorRoleCandidate, empty_role_reasons, select_actor_roles
import pytest
from step7.actor_roles import summarize_actor_role_rows

def candidate(track, x, y, relation='unknown', visible='shadow_visible', label='automobile'):
    return ActorRoleCandidate(track_id=track, label_class=label, geometry={'relative_x_m': x, 'relative_y_m': y, 'planar_distance_m': (x * x + y * y) ** 0.5, 'geometric_region': 'front'}, visibility={'shadow_status': visible, 'winning_cell_count': 3}, history={'history_status': 'usable', 'mean_distance_rate_mps': -1.0}, road={'ego_lane_relation': relation, 'lane_match_status': 'matched'})

def test_selects_three_distinct_roles_with_topology_priority():
    roles = select_actor_roles(candidates=(candidate('lead', 10, 0, 'same'), candidate('left', 2, 3, 'left_adjacent'), candidate('right', -2, -3, 'right_adjacent')))
    assert roles['lead_vehicle']['track_id'] == 'lead'
    assert roles['left_nearby_vehicle']['track_id'] == 'left'
    assert roles['right_nearby_vehicle']['track_id'] == 'right'

def test_invisible_and_nonvehicle_candidates_are_excluded():
    roles = select_actor_roles(candidates=(candidate('hidden', 5, 0, 'same', 'shadow_not_visible'), candidate('person', 5, 0, 'same', label='person')))
    assert all((value is None for value in roles.values()))

def test_history_quality_is_not_a_hard_gate():
    value = candidate('lead', 5, 0, 'same')
    value = ActorRoleCandidate(value.track_id, value.label_class, value.geometry, value.visibility, {'history_status': 'insufficient_span', 'mean_distance_rate_mps': None}, value.road)
    assert select_actor_roles(candidates=(value,))['lead_vehicle']['track_id'] == 'lead'

def test_empty_reason_distinguishes_no_actors_and_no_visible_vehicle():
    roles = select_actor_roles(candidates=())
    assert empty_role_reasons(candidates=(), roles=roles)['lead_vehicle'] == 'no_current_actors'
    values = (candidate('hidden', 5, 0, 'same', 'shadow_not_visible'),)
    roles = select_actor_roles(candidates=values)
    assert empty_role_reasons(candidates=values, roles=roles)['lead_vehicle'] == 'no_visible_vehicle_candidates'

def test_side_roles_reject_far_lateral_and_far_longitudinal_fallbacks():
    roles = select_actor_roles(candidates=(candidate('far_lateral', 2, 19, 'unknown'), candidate('far_forward', 42, -3, 'unrelated')))
    assert roles['left_nearby_vehicle'] is None
    assert roles['right_nearby_vehicle'] is None

def test_side_role_boundary_accepts_nearby_geometric_fallback():
    roles = select_actor_roles(candidates=(candidate('left', -10, 7.5, 'unknown'), candidate('right', 25, -6, 'unrelated')))
    assert roles['left_nearby_vehicle']['track_id'] == 'left'
    assert roles['right_nearby_vehicle']['track_id'] == 'right'

def row(anchor, duplicate=False):
    lead = {'track_id': '1'}
    return {'anchor_id': anchor, 'roles': {'lead_vehicle': lead, 'left_nearby_vehicle': lead if duplicate else None, 'right_nearby_vehicle': None}, 'empty_role_reasons': {'lead_vehicle': None, 'left_nearby_vehicle': None if duplicate else 'no_candidate_in_role_region', 'right_nearby_vehicle': 'no_candidate_in_role_region'}}

def test_summary_counts_roles_and_empty_reasons():
    result = summarize_actor_role_rows(keyframes=({'anchor_id': 'a'},), rows=(row('a'),))
    assert result['selected_role_counts']['lead_vehicle'] == 1
    assert result['role_conflict_count'] == 0

def test_duplicate_actor_across_roles_is_rejected():
    with pytest.raises(ValueError, match='multiple roles'):
        summarize_actor_role_rows(keyframes=({'anchor_id': 'a'},), rows=(row('a', duplicate=True),))
