from pathlib import Path


def test_build_road_context_writes_lane_direction_relation():
    source = Path('step7/build_road_context.py').read_text(encoding='utf-8')
    assert 'for track_id, label, match, relation, direction_relation in actor_matches' in source
    assert '"lane_direction_relation": direction_relation' in source
