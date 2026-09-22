"""Build and export the formal Step 7E geometric occlusion evidence product."""
from __future__ import annotations
import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from step7.occlusion import ActorGeometricOcclusionProjectionContext
from step7.occlusion import ActorGeometricOcclusionWithProjectionContextResult
import argparse
import time
from typing import Any
from project_paths import ALPASIM_DATA_ROOT, ANNOTATION_ROOT, MANIFEST_ROOT
from step2.clip_reader import DrivingClipReader
from step7.occlusion import CameraGeometricOcclusionBuildInput
from step7.occlusion import build_actor_geometric_occlusion_with_projection_context
from step7.projection import load_camera_calibration
from step7.scene_facts import CAMERA_NAMES
SCHEMA_VERSION = 'step7e-geometric-occlusion-evidence-v02'

@dataclass(frozen=True, slots=True)
class OcclusionExportInput:
    keyframe: Mapping[str, Any]
    context: ActorGeometricOcclusionProjectionContext
    is_static: bool

@dataclass(frozen=True, slots=True)
class AnchorCoverage:
    source_keyframe_count: int
    anchor_with_actor_rows_count: int
    anchor_without_actor_rows_count: int
    anchor_without_actor_rows: tuple[str, ...]
    rowless_anchor_reason_counts: tuple[tuple[str, int], ...]
    rowless_anchor_snapshot_evidence: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {'source_keyframe_count': self.source_keyframe_count, 'anchor_with_actor_rows_count': self.anchor_with_actor_rows_count, 'anchor_without_actor_rows_count': self.anchor_without_actor_rows_count, 'anchor_without_actor_rows': list(self.anchor_without_actor_rows), 'rowless_anchor_reason_counts': dict(self.rowless_anchor_reason_counts), 'rowless_anchor_snapshot_evidence': [dict(item) for item in self.rowless_anchor_snapshot_evidence]}

def _base_record(*, keyframe: Mapping[str, Any], context: ActorGeometricOcclusionProjectionContext, is_static: bool) -> dict[str, Any]:
    for field in ('anchor_id', 'clip_id', 'anchor_ns'):
        if field not in keyframe:
            raise ValueError(f'keyframe is missing {field}')
    evidence = context.combined_evidence
    if context.track_id != evidence.track_id:
        raise ValueError('projection context and combined evidence track_id values differ')
    if not evidence.track_id or not evidence.actor_class:
        raise ValueError('evidence identity fields must be non-empty')
    if evidence.static_occlusion_evaluated:
        raise ValueError('static occlusion must remain unevaluated in Step 7E v0.1')
    fraction = evidence.maximum_visible_fraction
    if fraction is not None:
        fraction = float(fraction)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError('maximum_visible_fraction must be finite and within [0, 1]')
    counts = (evidence.total_occupied_cell_count, evidence.total_winning_cell_count, evidence.total_occluded_cell_count)
    if any((isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts)):
        raise ValueError('combined cell counts must be non-negative integers')
    if counts[1] + counts[2] != counts[0]:
        raise ValueError('winning and occluded cell counts must close')
    projection_context = tuple(context.candidate_without_sampled_surface_projection_evidence)
    camera_names = tuple((item.camera_name for item in projection_context))
    if len(camera_names) != len(set(camera_names)):
        raise ValueError('projection context contains duplicate camera names')
    expected = tuple(evidence.geometric_candidate_without_sampled_surface_camera_names)
    if camera_names != expected:
        raise ValueError('projection-context cameras differ from missing-surface cameras')
    return {'schema_version': SCHEMA_VERSION, 'anchor_id': str(keyframe['anchor_id']), 'clip_id': str(keyframe['clip_id']), 'anchor_ns': int(keyframe['anchor_ns']), 'track_id': evidence.track_id, 'label_class': evidence.actor_class, 'is_static': bool(is_static), 'geometric_observability_status': evidence.geometric_observability_status, 'geometric_candidate_camera_names': list(evidence.geometric_candidate_camera_names), 'occlusion_evaluated_camera_names': list(evidence.occlusion_evaluated_camera_names), 'occlusion_winning_camera_names': list(evidence.occlusion_winning_camera_names), 'geometric_candidate_with_sampled_surface_camera_names': list(evidence.geometric_candidate_with_sampled_surface_camera_names), 'geometric_candidate_with_winning_cells_camera_names': list(evidence.geometric_candidate_with_winning_cells_camera_names), 'geometric_candidate_without_sampled_surface_camera_names': list(evidence.geometric_candidate_without_sampled_surface_camera_names), 'geometric_candidate_fully_occluded_camera_names': list(evidence.geometric_candidate_fully_occluded_camera_names), 'maximum_visible_fraction': fraction, 'total_occupied_cell_count': counts[0], 'total_winning_cell_count': counts[1], 'total_occluded_cell_count': counts[2], 'occluding_actor_ids': list(evidence.occluding_actor_ids), 'actor_to_actor_occlusion_evaluated': evidence.actor_to_actor_occlusion_evaluated, 'static_occlusion_evaluated': False, 'evidence_status': evidence.evidence_status, 'reasons': list(evidence.reasons), 'candidate_without_sampled_surface_projection_evidence': [item.to_dict() for item in projection_context]}

def encode_record(*, keyframe: Mapping[str, Any], context: ActorGeometricOcclusionProjectionContext, is_static: bool) -> str:
    return json.dumps(_base_record(keyframe=keyframe, context=context, is_static=is_static), ensure_ascii=False, separators=(',', ':'))

def prepare_export_inputs(*, keyframe: Mapping[str, Any], actors: Sequence[Mapping[str, Any]], result: ActorGeometricOcclusionWithProjectionContextResult) -> tuple[OcclusionExportInput, ...]:
    for field in ('anchor_id', 'clip_id', 'anchor_ns'):
        if field not in keyframe:
            raise ValueError(f'keyframe is missing {field}')
    source = tuple(actors)
    actors_by_id = {}
    for actor in source:
        track_id = str(actor.get('track_id', ''))
        if not track_id:
            raise ValueError('Actor is missing a usable track_id')
        if track_id in actors_by_id:
            raise ValueError('Actor snapshot contains duplicate track_id values')
        actors_by_id[track_id] = actor
    contexts = tuple(result.projection_context.actor_context)
    contexts_by_id = {item.track_id: item for item in contexts}
    if len(contexts_by_id) != len(contexts):
        raise ValueError('projection context contains duplicate track_id values')
    if result.actor_count != len(contexts) or result.projection_context.actor_count != len(contexts):
        raise ValueError('result actor_count and projection context count are inconsistent')
    if set(actors_by_id) != set(contexts_by_id):
        raise ValueError('Actor snapshot and projection-context sets must match')
    prepared = []
    for track_id in sorted(actors_by_id):
        actor = actors_by_id[track_id]
        if 'is_static' not in actor:
            raise ValueError(f'Actor {track_id} is missing is_static')
        context = contexts_by_id[track_id]
        if str(actor.get('label_class', '')) != context.combined_evidence.actor_class:
            raise ValueError(f'Actor class and projection context differ for track {track_id}')
        prepared.append(OcclusionExportInput(keyframe=keyframe, context=context, is_static=bool(actor['is_static'])))
    return tuple(prepared)

def _identity(item: OcclusionExportInput) -> tuple[str, str]:
    return (str(item.keyframe['anchor_id']), item.context.track_id)

def write_evidence(*, inputs: Sequence[OcclusionExportInput], output_path: Path, summary_path: Path) -> dict[str, Any]:
    source = tuple(inputs)
    identities = tuple((_identity(item) for item in source))
    if len(identities) != len(set(identities)):
        raise ValueError('duplicate anchor_id/track_id export keys')
    ordered = tuple(sorted(source, key=_identity))
    lines = tuple((encode_record(keyframe=item.keyframe, context=item.context, is_static=item.is_static) for item in ordered))
    output_text = ''.join((line + '\n' for line in lines))
    status_counts = Counter((item.context.combined_evidence.evidence_status for item in ordered))
    class_counts = Counter((item.context.combined_evidence.actor_class for item in ordered))
    projections = tuple((projection for item in ordered for projection in item.context.candidate_without_sampled_surface_projection_evidence))
    per_actor = Counter((len(item.context.candidate_without_sampled_surface_projection_evidence) for item in ordered))
    summary = {'schema_version': SCHEMA_VERSION, 'description': 'Threshold-free Step 7E geometric and Actor-to-Actor occlusion evidence with missing-surface projection context. No final visibility labels.', 'row_count': len(ordered), 'anchor_count': len({identity[0] for identity in identities}), 'evidence_status_counts': dict(sorted(status_counts.items())), 'rows_by_actor_class': dict(sorted(class_counts.items())), 'missing_surface_projection_context_count': len(projections), 'missing_surface_projection_context_status_counts': dict(sorted(Counter((item.evidence_status for item in projections)).items())), 'missing_surface_projection_context_camera_counts': dict(sorted(Counter((item.camera_name for item in projections)).items())), 'actors_with_missing_surface_projection_context_by_evidence_status': dict(sorted(Counter((item.context.combined_evidence.evidence_status for item in ordered if item.context.candidate_without_sampled_surface_projection_evidence)).items())), 'missing_surface_projection_context_parent_evidence_status_counts': dict(sorted(Counter((item.context.combined_evidence.evidence_status for item in ordered for _ in item.context.candidate_without_sampled_surface_projection_evidence)).items())), 'missing_surface_projection_context_count_per_actor_histogram': {str(key): value for key, value in sorted(per_actor.items())}, 'actor_to_actor_occlusion_evaluated_count': sum((item.context.combined_evidence.actor_to_actor_occlusion_evaluated for item in ordered)), 'static_occlusion_evaluated_count': sum((item.context.combined_evidence.static_occlusion_evaluated for item in ordered)), 'duplicate_key_count': 0, 'output_sha256': hashlib.sha256(output_text.encode('utf-8')).hexdigest()}
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output_path.with_suffix(output_path.suffix + '.tmp')
    summary_tmp = summary_path.with_suffix(summary_path.suffix + '.tmp')
    try:
        output_tmp.write_text(output_text, encoding='utf-8')
        summary_tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(output_tmp, output_path)
        os.replace(summary_tmp, summary_path)
    except Exception:
        output_tmp.unlink(missing_ok=True)
        summary_tmp.unlink(missing_ok=True)
        raise
    return summary

def summarize_anchor_coverage(*, keyframes: Sequence[Mapping[str, Any]], exported_identities: Sequence[tuple[str, str]], reader_factory: Callable[[str], Any]) -> AnchorCoverage:
    source = tuple(keyframes)
    anchor_ids = tuple((str(item['anchor_id']) for item in source))
    if any((not value for value in anchor_ids)) or len(anchor_ids) != len(set(anchor_ids)):
        raise ValueError('source keyframe anchor_id values must be non-empty and unique')
    identities = tuple(((str(anchor), str(track)) for anchor, track in exported_identities))
    if any((not anchor or not track for anchor, track in identities)) or len(identities) != len(set(identities)):
        raise ValueError('exported Actor identities must be non-empty and unique')
    source_set = set(anchor_ids)
    exported_set = {anchor for anchor, _ in identities}
    if exported_set - source_set:
        raise ValueError('exported rows contain Anchors absent from source Keyframes')
    rowless = tuple(sorted(source_set - exported_set))
    by_id = {str(item['anchor_id']): item for item in source}
    reasons = Counter()
    evidence = []
    readers = {}
    for anchor in rowless:
        keyframe = by_id[anchor]
        clip_id = str(keyframe['clip_id'])
        anchor_ns = int(keyframe['anchor_ns'])
        reader = readers.setdefault(clip_id, reader_factory(clip_id))
        snapshots = reader.get_actor_snapshots(anchor_ns, duration_ns=0)
        if len(snapshots) != 1:
            reason, actor_count, detail = ('exact_snapshot_count_not_one', None, len(snapshots))
        elif snapshots[0].stamp_ns != anchor_ns:
            reason, actor_count, detail = ('snapshot_timestamp_not_exact', None, snapshots[0].stamp_ns)
        else:
            actors = snapshots[0].message.get('actors')
            if not isinstance(actors, list):
                reason, actor_count, detail = ('actors_field_not_list', None, type(actors).__name__)
            elif actors:
                reason, actor_count, detail = ('exact_snapshot_has_actors', len(actors), None)
            else:
                reason, actor_count, detail = ('exact_snapshot_empty_actor_list', 0, None)
        reasons[reason] += 1
        evidence.append({'anchor_id': anchor, 'clip_id': clip_id, 'anchor_ns': anchor_ns, 'reason': reason, 'actor_count': actor_count, 'detail': detail})
    return AnchorCoverage(len(source), len(exported_set), len(rowless), rowless, tuple(sorted(reasons.items())), tuple(evidence))

def attach_coverage(*, summary: Mapping[str, Any], coverage: AnchorCoverage) -> dict[str, Any]:
    result = dict(summary)
    if int(result['anchor_count']) != coverage.anchor_with_actor_rows_count:
        raise ValueError('summary anchor_count differs from Anchors with Actor rows')
    if coverage.anchor_with_actor_rows_count + coverage.anchor_without_actor_rows_count != coverage.source_keyframe_count:
        raise ValueError('source Anchor coverage counts do not close')
    if sum(dict(coverage.rowless_anchor_reason_counts).values()) != coverage.anchor_without_actor_rows_count:
        raise ValueError('rowless Anchor reason counts do not close')
    result.update(coverage.to_dict())
    return result
KEYFRAME_PATH = ANNOTATION_ROOT / 'keyframes.jsonl'
OUTPUT_PATH = ANNOTATION_ROOT / 'step7e_geometric_occlusion_evidence_v02.jsonl'
SUMMARY_PATH = ANNOTATION_ROOT / 'step7e_geometric_occlusion_evidence_v02.summary.json'
KEYFRAME_CONTRACT_PATH = MANIFEST_ROOT / 'keyframe_contract_v0.1.json'
RASTER_WIDTH = 480
RASTER_HEIGHT = 270
MAXIMUM_DEPTH = 8
MAXIMUM_BOUNDARY_EXTENT_PX = 4.0
NEAR_PLANE_M = 0.001
MAXIMUM_CHORD_ERROR_PX = 1.0
MAXIMUM_ADAPTIVE_DEPTH = 14
DEPTH_TOLERANCE_M = 1e-09

def read_keyframes(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open('r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ('anchor_id', 'clip_id', 'anchor_ns'):
                if field not in record:
                    raise ValueError(f'{path}:{line_number}: missing {field}')
            records.append(record)
    if not records:
        raise RuntimeError(f'No Keyframes found in {path}')
    identities = tuple((str(item['anchor_id']) for item in records))
    if len(identities) != len(set(identities)):
        raise ValueError('Keyframe anchor_id values must be unique')
    return sorted(records, key=lambda item: str(item['anchor_id']))

def validate_keyframe_contract(*, contract_path: Path, keyframe_path: Path, keyframes: list[dict[str, Any]]) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    if not isinstance(contract, dict):
        raise ValueError('Keyframe contract root must be an object')
    if str(contract.get('contract_version')) != '0.1':
        raise ValueError('unsupported Keyframe contract version')
    if contract.get('producer_step') != 5:
        raise ValueError('Keyframe contract producer_step must be 5')
    actual = hashlib.sha256(keyframe_path.read_bytes()).hexdigest()
    if actual != str(contract.get('keyframe_sha256', '')):
        raise ValueError('Keyframe SHA-256 mismatch')
    if len(keyframes) != int(contract['keyframe_count']):
        raise ValueError('Keyframe count mismatch')
    return contract

def exact_inputs(reader: DrivingClipReader, anchor_ns: int):
    ego = reader.get_recorded_ego_state_at_or_before(anchor_ns)
    if ego is None or ego.stamp_ns != anchor_ns:
        raise RuntimeError('Exact recorded Ego state unavailable')
    snapshots = reader.get_actor_snapshots(anchor_ns, duration_ns=0)
    if len(snapshots) != 1 or snapshots[0].stamp_ns != anchor_ns:
        raise RuntimeError('Exact Actor snapshot unavailable')
    actors = snapshots[0].message.get('actors')
    if not isinstance(actors, list):
        raise RuntimeError('Actor snapshot has no actors list')
    return (ego, actors)

def load_camera_inputs(reader: DrivingClipReader, anchor_id: str, anchor_ns: int):
    cameras = {}
    for camera_name in CAMERA_NAMES:
        exact = reader.camera_indexes[camera_name].exact(anchor_ns)
        if exact is None:
            raise RuntimeError(f'No exact camera frame for {anchor_id} {camera_name}')
        frame = exact.value
        calibration = load_camera_calibration(reader.clip_directory / 'calibration' / f'{camera_name}.json', camera_name=camera_name, source_width=frame.width, source_height=frame.height)
        if calibration.max_angle_rad is None:
            raise RuntimeError(f'Camera calibration has no max_angle_rad: {camera_name}')
        cameras[camera_name] = CameraGeometricOcclusionBuildInput(calibration, frame.width, frame.height)
    return cameras

def build_anchor_inputs(*, keyframe, reader):
    anchor_id = str(keyframe['anchor_id'])
    anchor_ns = int(keyframe['anchor_ns'])
    ego, actors = exact_inputs(reader, anchor_ns)
    result = build_actor_geometric_occlusion_with_projection_context(actors=actors, recorded_ego_message=ego.message, cameras=load_camera_inputs(reader, anchor_id, anchor_ns), raster_width=RASTER_WIDTH, raster_height=RASTER_HEIGHT, maximum_depth=MAXIMUM_DEPTH, maximum_boundary_extent_px=MAXIMUM_BOUNDARY_EXTENT_PX, samples_per_edge=None, near_plane_m=NEAR_PLANE_M, maximum_chord_error_px=MAXIMUM_CHORD_ERROR_PX, maximum_adaptive_depth=MAXIMUM_ADAPTIVE_DEPTH, depth_tolerance_m=DEPTH_TOLERANCE_M)
    return prepare_export_inputs(keyframe=keyframe, actors=actors, result=result)

def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def export(*, keyframe_path: Path, output_path: Path, summary_path: Path, keyframe_contract_path: Path):
    started = time.monotonic()
    keyframes = read_keyframes(keyframe_path)
    contract = validate_keyframe_contract(contract_path=keyframe_contract_path, keyframe_path=keyframe_path, keyframes=keyframes)
    prepared = []
    readers = {}
    for index, keyframe in enumerate(keyframes, start=1):
        clip_id = str(keyframe['clip_id'])
        reader = readers.setdefault(clip_id, DrivingClipReader(ALPASIM_DATA_ROOT / clip_id))
        prepared.extend(build_anchor_inputs(keyframe=keyframe, reader=reader))
        if index % 10 == 0 or index == len(keyframes):
            elapsed = time.monotonic() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(keyframes) - index) / rate if rate else 0.0
            print(f'[Step 7E] {index}/{len(keyframes)} | rows {len(prepared)} | elapsed {_duration(elapsed)} | rate {rate:.2f}/s | ETA {_duration(eta)}', flush=True)
    summary = write_evidence(inputs=prepared, output_path=output_path, summary_path=summary_path)
    coverage = summarize_anchor_coverage(keyframes=keyframes, exported_identities=tuple(((str(item.keyframe['anchor_id']), item.context.track_id) for item in prepared)), reader_factory=lambda clip_id: DrivingClipReader(ALPASIM_DATA_ROOT / clip_id))
    summary = attach_coverage(summary=summary, coverage=coverage)
    summary.update({'keyframe_contract': str(keyframe_contract_path), 'keyframe_count': int(contract['keyframe_count']), 'keyframe_sha256': str(contract['keyframe_sha256']), 'keyframe_contract_version': str(contract['contract_version'])})
    temporary = summary_path.with_suffix(summary_path.suffix + '.tmp')
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, summary_path)
    print('Evidence rows:', summary['row_count'])
    print('Evidence SHA-256:', summary['output_sha256'])
    print('PASS: latest Step 7E v02 evidence exported.')
    return summary

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyframes', type=Path, default=KEYFRAME_PATH)
    parser.add_argument('--output', type=Path, default=OUTPUT_PATH)
    parser.add_argument('--summary', type=Path, default=SUMMARY_PATH)
    parser.add_argument('--keyframe-contract', type=Path, default=KEYFRAME_CONTRACT_PATH)
    args = parser.parse_args()
    export(keyframe_path=args.keyframes.expanduser().resolve(), output_path=args.output.expanduser().resolve(), summary_path=args.summary.expanduser().resolve(), keyframe_contract_path=args.keyframe_contract.expanduser().resolve())
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
