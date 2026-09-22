"""Tests for the consolidated Step 7E occlusion build and export entrypoint."""
from __future__ import annotations
import hashlib
import json
from types import SimpleNamespace
import pytest
from step7 import build_occlusion as target

def build_write_contract(tmp_path, text, count, *, digest=None, version='0.1', producer=5):
    keyframes = tmp_path / 'keyframes.jsonl'
    contract = tmp_path / 'contract.json'
    keyframes.write_text(text, encoding='utf-8')
    contract.write_text(json.dumps({'contract_version': version, 'producer_step': producer, 'keyframe_count': count, 'keyframe_sha256': digest or hashlib.sha256(text.encode()).hexdigest()}), encoding='utf-8')
    return (contract, keyframes)

def build_rows():
    return [{'anchor_id': 'a', 'clip_id': 'clip', 'anchor_ns': 1}, {'anchor_id': 'b', 'clip_id': 'clip', 'anchor_ns': 2}]

def test_read_keyframes_sorts_and_rejects_duplicates(tmp_path):
    path = tmp_path / 'keyframes.jsonl'
    path.write_text(json.dumps({'anchor_id': 'b', 'clip_id': 'c', 'anchor_ns': 2}) + '\n' + json.dumps({'anchor_id': 'a', 'clip_id': 'c', 'anchor_ns': 1}) + '\n', encoding='utf-8')
    assert [x['anchor_id'] for x in target.read_keyframes(path)] == ['a', 'b']
    path.write_text(json.dumps(build_rows()[0]) + '\n' + json.dumps(build_rows()[0]) + '\n')
    with pytest.raises(ValueError, match='unique'):
        target.read_keyframes(path)

def test_keyframe_contract_accepts_latest_contract(tmp_path):
    text = '{"anchor_id":"a"}\n{"anchor_id":"b"}\n'
    contract, keyframes = build_write_contract(tmp_path, text, 2)
    value = target.validate_keyframe_contract(contract_path=contract, keyframe_path=keyframes, keyframes=build_rows())
    assert value['keyframe_count'] == 2

@pytest.mark.parametrize('changes, message', [({'digest': '0' * 64}, 'SHA-256'), ({'count': 3}, 'count mismatch'), ({'version': '9.9'}, 'unsupported'), ({'producer': 4}, 'producer_step')])
def test_keyframe_contract_rejects_invalid_values(tmp_path, changes, message):
    text = '{"anchor_id":"a"}\n{"anchor_id":"b"}\n'
    values = {'count': 2, 'digest': None, 'version': '0.1', 'producer': 5}
    values.update(changes)
    contract, keyframes = build_write_contract(tmp_path, text, **values)
    with pytest.raises(ValueError, match=message):
        target.validate_keyframe_contract(contract_path=contract, keyframe_path=keyframes, keyframes=build_rows())

def test_exact_inputs_requires_exact_ego():
    reader = SimpleNamespace(get_recorded_ego_state_at_or_before=lambda value: SimpleNamespace(stamp_ns=value - 1), get_actor_snapshots=lambda *args, **kwargs: ())
    with pytest.raises(RuntimeError, match='Exact recorded Ego'):
        target.exact_inputs(reader, 100)

def export_context(track='1', count=0):
    projections = tuple((SimpleNamespace(camera_name=('cross_left', 'cross_right')[i], evidence_status='missing_surface_with_truncated_projection', to_dict=lambda i=i: {'camera_name': ('cross_left', 'cross_right')[i], 'evidence_status': 'missing_surface_with_truncated_projection'}) for i in range(count)))
    missing = tuple((export_item.camera_name for export_item in projections))
    evidence = SimpleNamespace(track_id=track, actor_class='automobile', static_occlusion_evaluated=False, maximum_visible_fraction=None, total_occupied_cell_count=0, total_winning_cell_count=0, total_occluded_cell_count=0, geometric_observability_status='candidate_visible', geometric_candidate_camera_names=missing, occlusion_evaluated_camera_names=(), occlusion_winning_camera_names=(), geometric_candidate_with_sampled_surface_camera_names=(), geometric_candidate_with_winning_cells_camera_names=(), geometric_candidate_without_sampled_surface_camera_names=missing, geometric_candidate_fully_occluded_camera_names=(), occluding_actor_ids=(), actor_to_actor_occlusion_evaluated=True, evidence_status='candidate_without_sampled_surface', reasons=())
    return SimpleNamespace(track_id=track, combined_evidence=evidence, candidate_without_sampled_surface_projection_evidence=projections)

def export_item(anchor, track, count=0):
    return target.OcclusionExportInput({'anchor_id': anchor, 'clip_id': 'clip', 'anchor_ns': 1}, export_context(track, count), False)

def test_v02_record_has_only_latest_schema():
    row = json.loads(target.encode_record(keyframe=export_item('a', '1', 1).keyframe, context=export_context('1', 1), is_static=False))
    assert row['schema_version'] == 'step7e-geometric-occlusion-evidence-v02'
    assert len(row['candidate_without_sampled_surface_projection_evidence']) == 1

def test_writer_is_sorted_deterministic_and_hashed(tmp_path):
    output = tmp_path / 'evidence.jsonl'
    summary_path = tmp_path / 'summary.json'
    summary = target.write_evidence(inputs=(export_item('b', '2'), export_item('a', '9'), export_item('a', '1', 1)), output_path=output, summary_path=summary_path)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(row['anchor_id'], row['track_id']) for row in rows] == [('a', '1'), ('a', '9'), ('b', '2')]
    assert summary['output_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()

def test_duplicate_identity_is_rejected(tmp_path):
    value = export_item('a', '1')
    with pytest.raises(ValueError, match='duplicate'):
        target.write_evidence(inputs=(value, value), output_path=tmp_path / 'x', summary_path=tmp_path / 's')

def test_prepare_inputs_joins_latest_context_only():
    value = export_context('1')
    result = SimpleNamespace(actor_count=1, projection_context=SimpleNamespace(actor_count=1, actor_context=(value,)))
    prepared = target.prepare_export_inputs(keyframe={'anchor_id': 'a', 'clip_id': 'c', 'anchor_ns': 1}, actors=({'track_id': '1', 'label_class': 'automobile', 'is_static': False},), result=result)
    assert prepared[0].context is value
