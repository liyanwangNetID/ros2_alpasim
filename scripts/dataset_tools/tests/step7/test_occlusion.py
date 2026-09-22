"""Tests for the consolidated Step 7 occlusion domain."""
from __future__ import annotations
from dataclasses import dataclass
from types import SimpleNamespace
import pytest
from step7 import occlusion as target
from step7.occlusion import build_camera_actor_occlusion_pipeline
from step7.raster import ActorDepthRasterInput
from dataclasses import replace
from step7.occlusion import build_actor_geometric_occlusion_evidence_set
from step7.occlusion import ActorMulticameraOcclusionSummary
from step7.observability import ActorObservability, CameraObservability
from step7.scene_facts import CAMERA_NAMES, OBSERVABILITY_FORMAT_VERSION
from step7.occlusion import summarize_actor_geometric_occlusion_evidence
from step7.occlusion import ActorGeometricOcclusionEvidence
from step7.occlusion import build_actor_geometric_occlusion_evidence
from step7.scene_facts import CAMERA_NAMES
from step7.occlusion import build_actor_geometric_occlusion_pipeline
from step7.occlusion import summarize_actor_multicamera_occlusion
from step7.occlusion import build_camera_actor_occlusion_evidence
from step7.raster import ActorDepthRasterInput, ActorDepthZBuffer, ActorZBufferSummary, ZBufferCellWinner
from step7.occlusion import build_camera_occlusion_evidence_from_zbuffer
from step7.occlusion import OCCLUSION_EVIDENCE_FORMAT_VERSION, build_camera_actor_occlusion_evidence
from step7.projection import ActorCameraProjection
from step7.occlusion import build_candidate_without_sampled_surface_projection_evidence

@dataclass(frozen=True)
class Cell:
    column: int
    row: int

@dataclass(frozen=True)
class Sample:
    cell: Cell
    depth_m: float

@dataclass(frozen=True)
class Raster:
    cell_depths: tuple[Sample, ...]
    occupied_cell_count: int

class Calibration:
    max_angle_rad = 1.0

    def rig_point_to_camera(self, point):
        return ('camera', point)

def cameraresult_actor(track_id):
    return {'track_id': track_id}

def cameraresult_install_geometry_stubs(monkeypatch):
    monkeypatch.setattr(target, 'actor_box_corners_in_rig', lambda value, recorded_ego_message: (value['track_id'],))
    monkeypatch.setattr(target, 'prepare_camera_facing_box_triangles', lambda corners, near_plane_m: (SimpleNamespace(vertices_camera=corners),))

    def build_surface(triangles, calibration, **kwargs):
        track_id = triangles[0][0][1]
        if track_id == 'near':
            samples = (Sample(Cell(0, 0), 5.0), Sample(Cell(1, 0), 5.0))
        elif track_id == 'far':
            samples = (Sample(Cell(0, 0), 10.0), Sample(Cell(1, 0), 10.0), Sample(Cell(2, 0), 10.0))
        else:
            samples = ()
        return SimpleNamespace(surface_raster=Raster(samples, len(samples)), generated_triangle_sample_count=1 if samples else 0, zero_center_sample_triangle_count=0 if samples else 1, unresolved_boundary_count=0)
    monkeypatch.setattr(target, 'build_actor_camera_surface_depth_raster', build_surface)

def cameraresult_call(monkeypatch, **overrides):
    cameraresult_install_geometry_stubs(monkeypatch)
    values = {'actors': (cameraresult_actor('near'), cameraresult_actor('far'), cameraresult_actor('empty')), 'recorded_ego_message': {'ego': True}, 'calibration': Calibration(), 'camera_name': 'cross_left', 'image_width_px': 1920, 'image_height_px': 1080, 'raster_width': 480, 'raster_height': 270, 'maximum_depth': 8, 'maximum_boundary_extent_px': 4.0}
    values.update(overrides)
    return target.build_actor_camera_occlusion_from_geometry(**values)

def test_builds_surface_diagnostics_and_shared_evidence(monkeypatch):
    result = cameraresult_call(monkeypatch)
    evidence = {item.track_id: item for item in result.occlusion.actor_evidence}
    assert result.actor_count == 3
    assert evidence['far'].occupied_cell_count == 3
    assert evidence['far'].winning_cell_count == 1
    assert evidence['far'].occluding_actor_ids == ('near',)
    assert evidence['near'].visible_fraction == 1.0
    assert evidence['empty'].evidence_status == 'no_sampled_surface'

def test_output_is_deterministic_by_track_id(monkeypatch):
    result = cameraresult_call(monkeypatch, actors=(cameraresult_actor('near'), cameraresult_actor('empty'), cameraresult_actor('far')))
    assert tuple((item.track_id for item in result.surface_diagnostics)) == ('empty', 'far', 'near')
    assert tuple((item.track_id for item in result.occlusion.actor_evidence)) == ('empty', 'far', 'near')

def test_surface_diagnostics_preserve_zero_surface(monkeypatch):
    result = cameraresult_call(monkeypatch)
    diagnostics = {item.track_id: item for item in result.surface_diagnostics}
    assert diagnostics['empty'].occupied_cell_count == 0
    assert diagnostics['empty'].generated_triangle_sample_count == 0
    assert diagnostics['empty'].zero_center_sample_triangle_count == 1
    assert diagnostics['empty'].unresolved_boundary_count == 0

def test_result_metadata_matches_request(monkeypatch):
    result = cameraresult_call(monkeypatch)
    assert result.camera_name == 'cross_left'
    assert result.raster_width == 480
    assert result.raster_height == 270
    assert result.occlusion.camera_name == 'cross_left'

def test_duplicate_track_ids_are_rejected(monkeypatch):
    cameraresult_install_geometry_stubs(monkeypatch)
    with pytest.raises(ValueError, match='must be unique'):
        cameraresult_call(monkeypatch, actors=(cameraresult_actor('near'), cameraresult_actor('near')))

def test_missing_track_id_is_rejected(monkeypatch):
    cameraresult_install_geometry_stubs(monkeypatch)
    with pytest.raises(ValueError, match='usable track_id'):
        cameraresult_call(monkeypatch, actors=({},))

def test_missing_calibration_angle_is_rejected(monkeypatch):
    cameraresult_install_geometry_stubs(monkeypatch)
    calibration = Calibration()
    calibration.max_angle_rad = None
    with pytest.raises(ValueError, match='max_angle_rad'):
        cameraresult_call(monkeypatch, calibration=calibration)

def test_static_occlusion_remains_false(monkeypatch):
    result = cameraresult_call(monkeypatch)
    assert all((not item.static_occlusion_evaluated for item in result.occlusion.actor_evidence))

@dataclass(frozen=True)
class Cell_2:
    column: int
    row: int

@dataclass(frozen=True)
class SurfaceSample:
    cell: Cell_2
    depth_m: float

@dataclass(frozen=True)
class SurfaceRaster:
    cell_depths: tuple[SurfaceSample, ...]
    occupied_cell_count: int

def camerapipeline_raster(*cells):
    samples = tuple((SurfaceSample(Cell_2(column, row), depth) for column, row, depth in cells))
    return SurfaceRaster(samples, len(samples))

def camerapipeline_inputs():
    return (ActorDepthRasterInput('near', camerapipeline_raster((0, 0, 5.0), (1, 0, 5.0))), ActorDepthRasterInput('far', camerapipeline_raster((0, 0, 10.0), (1, 0, 10.0), (2, 0, 10.0))), ActorDepthRasterInput('empty', camerapipeline_raster()))

def camerapipeline_call(**overrides):
    values = {'camera_name': 'cross_left', 'raster_width': 480, 'raster_height': 270, 'actor_rasters': camerapipeline_inputs(), 'depth_tolerance_m': 1e-09, 'static_occlusion_evaluated': False}
    values.update(overrides)
    return build_camera_actor_occlusion_pipeline(**values)

def test_pipeline_resolves_zbuffer_and_evidence():
    result = camerapipeline_call()
    evidence = {item.track_id: item for item in result.actor_evidence}
    assert result.zbuffer.occupied_union_cell_count == 3
    assert result.zbuffer.contested_cell_count == 2
    assert evidence['far'].occupied_cell_count == 3
    assert evidence['far'].winning_cell_count == 1
    assert evidence['far'].occluded_cell_count == 2
    assert evidence['far'].visible_fraction == pytest.approx(1 / 3)
    assert evidence['far'].occluding_actor_ids == ('near',)
    assert evidence['near'].visible_fraction == 1.0
    assert evidence['near'].occluding_actor_ids == ()

def test_pipeline_preserves_zero_surface_actor():
    result = camerapipeline_call()
    evidence = {item.track_id: item for item in result.actor_evidence}
    assert evidence['empty'].evidence_status == 'no_sampled_surface'
    assert evidence['empty'].visible_fraction is None
    assert evidence['empty'].occluding_actor_ids == ()

def test_camerapipeline_result_metadata_matches_request():
    result = camerapipeline_call()
    assert result.camera_name == 'cross_left'
    assert result.raster_width == 480
    assert result.raster_height == 270
    assert tuple((item.track_id for item in result.actor_evidence)) == ('empty', 'far', 'near')

def test_static_occlusion_flag_is_forwarded_without_evaluation_logic():
    result = camerapipeline_call(static_occlusion_evaluated=True)
    assert all((item.static_occlusion_evaluated for item in result.actor_evidence))

def test_duplicate_actor_ids_are_rejected_by_shared_zbuffer():
    source = camerapipeline_inputs()
    with pytest.raises(ValueError, match='must be unique'):
        camerapipeline_call(actor_rasters=(source[0], source[0]))

def test_invalid_depth_tolerance_is_rejected():
    with pytest.raises(ValueError, match='non-negative and finite'):
        camerapipeline_call(depth_tolerance_m=-1.0)

def test_invalid_raster_dimensions_are_rejected_by_evidence_contract():
    with pytest.raises(ValueError, match='must be positive'):
        camerapipeline_call(raster_width=0)

def evidenceset_geometric(track_id, actor_class='automobile'):
    return ActorObservability(observability_format_version=OBSERVABILITY_FORMAT_VERSION, track_id=track_id, actor_class=actor_class, observability_status='candidate_visible', visible_in_cameras=(CAMERA_NAMES[0],), camera_observability=tuple((CameraObservability(camera_name=name, projection_valid=True, geometric_observability_candidate=name == CAMERA_NAMES[0], failure_reason=None if name == CAMERA_NAMES[0] else 'below') for name in CAMERA_NAMES)), actor_to_actor_occlusion_evaluated=False, static_occlusion_evaluated=False)

def evidenceset_occlusion(track_id):
    return ActorMulticameraOcclusionSummary(track_id=track_id, evaluated_camera_names=(CAMERA_NAMES[0],), no_sampled_surface_camera_names=tuple(CAMERA_NAMES[1:]), winning_camera_names=(CAMERA_NAMES[0],), fully_occluded_camera_names=(), occluded_camera_names=(), evaluated_camera_count=1, winning_camera_count=1, total_occupied_cell_count=10, total_winning_cell_count=10, total_occluded_cell_count=0, maximum_visible_fraction=1.0, occluding_actor_ids=(), actor_to_actor_occlusion_evaluated=True, static_occlusion_evaluated=False, reasons=())

def test_joins_complete_actor_sets_in_track_order():
    result = build_actor_geometric_occlusion_evidence_set(geometric_observability=(evidenceset_geometric('2'), evidenceset_geometric('1')), occlusion_summaries=(evidenceset_occlusion('1'), evidenceset_occlusion('2')))
    assert result.actor_count == 2
    assert tuple((item.track_id for item in result.actor_evidence)) == ('1', '2')
    assert all((item.evidence_status == 'combined_evidence_available' for item in result.actor_evidence))

def test_combined_evidence_preserves_actor_class():
    result = build_actor_geometric_occlusion_evidence_set(geometric_observability=(evidenceset_geometric('1', 'person'),), occlusion_summaries=(evidenceset_occlusion('1'),))
    assert result.actor_evidence[0].actor_class == 'person'

def test_geometric_duplicate_track_id_is_rejected():
    with pytest.raises(ValueError, match='geometric observability contains duplicate'):
        build_actor_geometric_occlusion_evidence_set(geometric_observability=(evidenceset_geometric('1'), evidenceset_geometric('1')), occlusion_summaries=(evidenceset_occlusion('1'),))

def test_occlusion_duplicate_track_id_is_rejected():
    with pytest.raises(ValueError, match='occlusion summaries contain duplicate'):
        build_actor_geometric_occlusion_evidence_set(geometric_observability=(evidenceset_geometric('1'),), occlusion_summaries=(evidenceset_occlusion('1'), evidenceset_occlusion('1')))

def test_actor_set_mismatch_is_rejected():
    with pytest.raises(ValueError, match='Actor sets must match'):
        build_actor_geometric_occlusion_evidence_set(geometric_observability=(evidenceset_geometric('1'), evidenceset_geometric('2')), occlusion_summaries=(evidenceset_occlusion('1'), evidenceset_occlusion('3')))

def test_empty_sets_produce_empty_result():
    result = build_actor_geometric_occlusion_evidence_set(geometric_observability=(), occlusion_summaries=())
    assert result.actor_count == 0
    assert result.actor_evidence == ()

def evidencesummary_item(track_id, *, status='combined_evidence_available', candidates=('front_wide',), sampled=('front_wide',), winners=('front_wide',), without_surface=(), fully_occluded=(), occluders=(), complete=True, static=False):
    return ActorGeometricOcclusionEvidence(track_id=track_id, actor_class='automobile', geometric_observability_status='candidate_visible' if candidates else 'not_visible', geometric_candidate_camera_names=candidates, occlusion_evaluated_camera_names=('front_wide',), occlusion_winning_camera_names=winners, geometric_candidate_with_sampled_surface_camera_names=sampled, geometric_candidate_with_winning_cells_camera_names=winners, geometric_candidate_without_sampled_surface_camera_names=without_surface, geometric_candidate_fully_occluded_camera_names=fully_occluded, actor_to_actor_occlusion_evaluated=complete, static_occlusion_evaluated=static, maximum_visible_fraction=1.0 if sampled else None, total_occupied_cell_count=10 if sampled else 0, total_winning_cell_count=10 if winners else 0, total_occluded_cell_count=0 if winners else 10 if sampled else 0, occluding_actor_ids=occluders, evidence_status=status, reasons=())

def evidencesummary_source():
    return (evidencesummary_item('1'), evidencesummary_item('2', winners=(), fully_occluded=('front_wide',), occluders=('9',)), evidencesummary_item('3', status='candidate_without_sampled_surface', sampled=(), winners=(), without_surface=('front_wide',)), evidencesummary_item('4', status='no_geometric_candidate', candidates=(), sampled=(), winners=()))

def test_summarizes_combined_evidence_counts():
    result = summarize_actor_geometric_occlusion_evidence(evidencesummary_source())
    assert result.actor_count == 4
    assert dict(result.evidence_status_counts) == {'candidate_without_sampled_surface': 1, 'combined_evidence_available': 2, 'no_geometric_candidate': 1}
    assert result.geometric_candidate_actor_count == 3
    assert result.geometric_candidate_with_sampled_surface_actor_count == 2
    assert result.geometric_candidate_with_winning_cells_actor_count == 1
    assert result.geometric_candidate_without_sampled_surface_actor_count == 1
    assert result.geometric_candidate_fully_occluded_actor_count == 1
    assert result.actor_with_occluder_count == 1
    assert result.actor_to_actor_occlusion_complete_count == 4
    assert result.static_occlusion_evaluated_count == 0

def test_preserves_review_track_ids():
    result = summarize_actor_geometric_occlusion_evidence(evidencesummary_source())
    assert result.geometric_candidate_without_sampled_surface_track_ids == ('3',)
    assert result.geometric_candidate_fully_occluded_track_ids == ('2',)

def test_incomplete_occlusion_is_reported():
    values = list(evidencesummary_source())
    values[0] = replace(values[0], actor_to_actor_occlusion_evaluated=False)
    result = summarize_actor_geometric_occlusion_evidence(values)
    assert result.actor_to_actor_occlusion_complete_count == 3
    assert result.reasons == ('one_or_more_actors_have_incomplete_occlusion_evidence',)

def test_static_occlusion_presence_is_reported():
    values = list(evidencesummary_source())
    values[0] = replace(values[0], static_occlusion_evaluated=True)
    result = summarize_actor_geometric_occlusion_evidence(values)
    assert result.static_occlusion_evaluated_count == 1
    assert result.reasons == ('one_or_more_actors_have_static_occlusion_evidence',)

def test_evidencesummary_duplicate_track_ids_are_rejected():
    first = evidencesummary_source()[0]
    with pytest.raises(ValueError, match='must be unique'):
        summarize_actor_geometric_occlusion_evidence((first, first))

def test_empty_input_produces_empty_summary():
    result = summarize_actor_geometric_occlusion_evidence(())
    assert result.actor_count == 0
    assert result.evidence_status_counts == ()
    assert result.reasons == ()

def test_to_dict_uses_mapping_and_lists():
    value = summarize_actor_geometric_occlusion_evidence(evidencesummary_source()).to_dict()
    assert isinstance(value['evidence_status_counts'], dict)
    assert isinstance(value['geometric_candidate_without_sampled_surface_track_ids'], list)
    assert isinstance(value['reasons'], list)

def evidence_geometric(*candidate_cameras):
    return ActorObservability(observability_format_version=OBSERVABILITY_FORMAT_VERSION, track_id='13', actor_class='automobile', observability_status='candidate_visible' if candidate_cameras else 'not_visible', visible_in_cameras=tuple(candidate_cameras), camera_observability=tuple((CameraObservability(camera_name=name, projection_valid=True, geometric_observability_candidate=name in candidate_cameras, failure_reason=None if name in candidate_cameras else 'below') for name in CAMERA_NAMES)), actor_to_actor_occlusion_evaluated=False, static_occlusion_evaluated=False)

def evidence_occlusion(**overrides):
    values = {'track_id': '13', 'evaluated_camera_names': tuple(CAMERA_NAMES[:3]), 'no_sampled_surface_camera_names': (CAMERA_NAMES[3],), 'winning_camera_names': (CAMERA_NAMES[0], CAMERA_NAMES[2]), 'fully_occluded_camera_names': (CAMERA_NAMES[1],), 'occluded_camera_names': (CAMERA_NAMES[1], CAMERA_NAMES[2]), 'evaluated_camera_count': 3, 'winning_camera_count': 2, 'total_occupied_cell_count': 30, 'total_winning_cell_count': 14, 'total_occluded_cell_count': 16, 'maximum_visible_fraction': 1.0, 'occluding_actor_ids': ('18', '29'), 'actor_to_actor_occlusion_evaluated': True, 'static_occlusion_evaluated': False, 'reasons': ()}
    values.update(overrides)
    return ActorMulticameraOcclusionSummary(**values)

def test_combines_geometric_candidates_with_occlusion_camera_sets():
    result = build_actor_geometric_occlusion_evidence(geometric=evidence_geometric(CAMERA_NAMES[0], CAMERA_NAMES[1], CAMERA_NAMES[3]), occlusion=evidence_occlusion())
    assert result.evidence_status == 'combined_evidence_available'
    assert result.geometric_candidate_with_sampled_surface_camera_names == (CAMERA_NAMES[0], CAMERA_NAMES[1])
    assert result.geometric_candidate_with_winning_cells_camera_names == (CAMERA_NAMES[0],)
    assert result.geometric_candidate_without_sampled_surface_camera_names == (CAMERA_NAMES[3],)
    assert result.geometric_candidate_fully_occluded_camera_names == (CAMERA_NAMES[1],)
    assert result.reasons == ()

def test_no_geometric_candidate_is_explicit():
    result = build_actor_geometric_occlusion_evidence(geometric=evidence_geometric(), occlusion=evidence_occlusion())
    assert result.evidence_status == 'no_geometric_candidate'
    assert result.reasons == ('no_geometric_candidate_camera',)

def test_candidate_without_surface_is_explicit():
    result = build_actor_geometric_occlusion_evidence(geometric=evidence_geometric(CAMERA_NAMES[3]), occlusion=evidence_occlusion())
    assert result.evidence_status == 'candidate_without_sampled_surface'
    assert result.geometric_candidate_without_sampled_surface_camera_names == (CAMERA_NAMES[3],)
    assert result.reasons == ('no_geometric_candidate_has_sampled_surface',)

def test_incomplete_occlusion_has_precedence():
    result = build_actor_geometric_occlusion_evidence(geometric=evidence_geometric(CAMERA_NAMES[0]), occlusion=evidence_occlusion(actor_to_actor_occlusion_evaluated=False))
    assert result.evidence_status == 'occlusion_incomplete'
    assert result.reasons == ('actor_to_actor_occlusion_not_fully_evaluated',)

def test_continuous_and_occluder_values_are_preserved():
    result = build_actor_geometric_occlusion_evidence(geometric=evidence_geometric(CAMERA_NAMES[0]), occlusion=evidence_occlusion())
    assert result.maximum_visible_fraction == 1.0
    assert result.total_occupied_cell_count == 30
    assert result.total_winning_cell_count == 14
    assert result.total_occluded_cell_count == 16
    assert result.occluding_actor_ids == ('18', '29')

def test_track_mismatch_is_rejected():
    with pytest.raises(ValueError, match='track_id values must match'):
        build_actor_geometric_occlusion_evidence(geometric=evidence_geometric(CAMERA_NAMES[0]), occlusion=replace(evidence_occlusion(), track_id='other'))

def test_noncanonical_geometric_order_is_rejected():
    with pytest.raises(ValueError, match='canonical order'):
        build_actor_geometric_occlusion_evidence(geometric=evidence_geometric(CAMERA_NAMES[1], CAMERA_NAMES[0]), occlusion=evidence_occlusion())

def test_to_dict_uses_json_ready_lists():
    value = build_actor_geometric_occlusion_evidence(geometric=evidence_geometric(CAMERA_NAMES[0]), occlusion=evidence_occlusion()).to_dict()
    assert isinstance(value['geometric_candidate_camera_names'], list)
    assert isinstance(value['occlusion_winning_camera_names'], list)
    assert isinstance(value['occluding_actor_ids'], list)
    assert isinstance(value['reasons'], list)

def fromgeometry_cameras():
    return {name: target.CameraGeometricOcclusionBuildInput(calibration=SimpleNamespace(camera_name=name), image_width_px=1920, image_height_px=1080) for name in CAMERA_NAMES}

def fromgeometry_install_stubs(monkeypatch):
    calls = []

    def geometric(**kwargs):
        calls.append(('geometric', kwargs))
        values = tuple((SimpleNamespace(track_id=str(actor['track_id'])) for actor in sorted(kwargs['actors'], key=lambda item: str(item['track_id']))))
        return SimpleNamespace(actor_count=len(values), actor_observability=values)

    def occlusion(**kwargs):
        calls.append(('occlusion', kwargs))
        values = tuple((SimpleNamespace(track_id=str(actor['track_id'])) for actor in sorted(kwargs['actors'], key=lambda item: str(item['track_id']))))
        return SimpleNamespace(actor_count=len(values), actor_summaries=values, actor_evidence=values)

    def combine(*, geometric_observability, multicamera_occlusion):
        calls.append(('combine', {'geometric_observability': geometric_observability, 'multicamera_occlusion': multicamera_occlusion}))
        evidence = SimpleNamespace(actor_evidence=geometric_observability)
        return SimpleNamespace(actor_count=len(geometric_observability), combined_evidence=evidence)
    monkeypatch.setattr(target, 'build_multicamera_actor_geometric_observability', geometric)
    monkeypatch.setattr(target, 'build_multicamera_actor_occlusion_from_geometry', occlusion)
    monkeypatch.setattr(target, 'build_actor_geometric_occlusion_pipeline', combine)
    return calls

def fromgeometry_call(monkeypatch, **overrides):
    calls = fromgeometry_install_stubs(monkeypatch)
    values = {'actors': ({'track_id': '2', 'label_class': 'automobile'}, {'track_id': '1', 'label_class': 'person'}), 'recorded_ego_message': {'ego': True}, 'cameras': fromgeometry_cameras(), 'raster_width': 480, 'raster_height': 270, 'maximum_depth': 8, 'maximum_boundary_extent_px': 4.0}
    values.update(overrides)
    return (target.build_actor_geometric_occlusion_from_geometry(**values), calls)

def test_runs_geometric_occlusion_and_combination(monkeypatch):
    result, calls = fromgeometry_call(monkeypatch)
    assert result.actor_count == 2
    assert tuple((name for name, _ in calls)) == ('geometric', 'occlusion', 'combine')
    assert tuple((item.track_id for item in result.combined.combined_evidence.actor_evidence)) == ('1', '2')

def test_uses_same_calibration_objects_for_both_paths(monkeypatch):
    result, calls = fromgeometry_call(monkeypatch)
    assert result.actor_count == 2
    geometric_call = calls[0][1]
    occlusion_call = calls[1][1]
    for name in CAMERA_NAMES:
        assert geometric_call['calibrations'][name] is occlusion_call['cameras'][name].calibration

def test_parameters_are_forwarded_to_correct_paths(monkeypatch):
    _, calls = fromgeometry_call(monkeypatch, samples_per_edge=9, near_plane_m=0.01, maximum_chord_error_px=0.5, maximum_adaptive_depth=11, depth_tolerance_m=1e-08)
    geometric_call = calls[0][1]
    occlusion_call = calls[1][1]
    assert geometric_call['samples_per_edge'] == 9
    assert geometric_call['maximum_chord_error_px'] == 0.5
    assert geometric_call['maximum_adaptive_depth'] == 11
    assert geometric_call['near_plane_m'] == 0.01
    assert occlusion_call['near_plane_m'] == 0.01
    assert occlusion_call['depth_tolerance_m'] == 1e-08

def test_camera_dimensions_are_forwarded_to_occlusion_path(monkeypatch):
    _, calls = fromgeometry_call(monkeypatch)
    occlusion_call = calls[1][1]
    for name in CAMERA_NAMES:
        camera = occlusion_call['cameras'][name]
        assert camera.image_width_px == 1920
        assert camera.image_height_px == 1080

def test_missing_camera_is_rejected(monkeypatch):
    source = fromgeometry_cameras()
    del source[CAMERA_NAMES[0]]
    with pytest.raises(ValueError, match='missing'):
        fromgeometry_call(monkeypatch, cameras=source)

def test_extra_camera_is_rejected(monkeypatch):
    source = fromgeometry_cameras()
    source['extra'] = target.CameraGeometricOcclusionBuildInput(calibration=object(), image_width_px=1, image_height_px=1)
    with pytest.raises(ValueError, match='extra'):
        fromgeometry_call(monkeypatch, cameras=source)

def test_empty_actor_set_is_supported(monkeypatch):
    result, _ = fromgeometry_call(monkeypatch, actors=())
    assert result.actor_count == 0
    assert result.combined.combined_evidence.actor_evidence == ()

def pipeline_geometric(track_id):
    return ActorObservability(observability_format_version=OBSERVABILITY_FORMAT_VERSION, track_id=track_id, actor_class='automobile', observability_status='candidate_visible', visible_in_cameras=(CAMERA_NAMES[0],), camera_observability=tuple((CameraObservability(camera_name=name, projection_valid=True, geometric_observability_candidate=name == CAMERA_NAMES[0], failure_reason=None if name == CAMERA_NAMES[0] else 'below') for name in CAMERA_NAMES)), actor_to_actor_occlusion_evaluated=False, static_occlusion_evaluated=False)

def pipeline_summary(track_id):
    return ActorMulticameraOcclusionSummary(track_id=track_id, evaluated_camera_names=(CAMERA_NAMES[0],), no_sampled_surface_camera_names=tuple(CAMERA_NAMES[1:]), winning_camera_names=(CAMERA_NAMES[0],), fully_occluded_camera_names=(), occluded_camera_names=(), evaluated_camera_count=1, winning_camera_count=1, total_occupied_cell_count=10, total_winning_cell_count=10, total_occluded_cell_count=0, maximum_visible_fraction=1.0, occluding_actor_ids=(), actor_to_actor_occlusion_evaluated=True, static_occlusion_evaluated=False, reasons=())

def pipeline_multicamera(track_ids=('1', '2')):
    summaries = tuple((pipeline_summary(track_id) for track_id in track_ids))
    evidence = tuple((SimpleNamespace(track_id=track_id) for track_id in track_ids))
    return SimpleNamespace(actor_count=len(track_ids), actor_summaries=summaries, actor_evidence=evidence)

def test_composes_complete_results_in_track_order():
    result = build_actor_geometric_occlusion_pipeline(geometric_observability=(pipeline_geometric('2'), pipeline_geometric('1')), multicamera_occlusion=pipeline_multicamera())
    assert result.actor_count == 2
    assert tuple((item.track_id for item in result.geometric_observability)) == ('1', '2')
    assert tuple((item.track_id for item in result.combined_evidence.actor_evidence)) == ('1', '2')

def test_preserves_multicamera_result_object():
    source = pipeline_multicamera()
    result = build_actor_geometric_occlusion_pipeline(geometric_observability=(pipeline_geometric('1'), pipeline_geometric('2')), multicamera_occlusion=source)
    assert result.multicamera_occlusion is source

def test_multicamera_summary_count_mismatch_is_rejected():
    source = pipeline_multicamera()
    source.actor_summaries = source.actor_summaries[:1]
    with pytest.raises(ValueError, match='actor_count and summaries'):
        build_actor_geometric_occlusion_pipeline(geometric_observability=(pipeline_geometric('1'), pipeline_geometric('2')), multicamera_occlusion=source)

def test_multicamera_evidence_count_mismatch_is_rejected():
    source = pipeline_multicamera()
    source.actor_evidence = source.actor_evidence[:1]
    with pytest.raises(ValueError, match='actor_count and evidence'):
        build_actor_geometric_occlusion_pipeline(geometric_observability=(pipeline_geometric('1'), pipeline_geometric('2')), multicamera_occlusion=source)

def test_pipeline_actor_set_mismatch_is_rejected():
    with pytest.raises(ValueError, match='Actor sets must match'):
        build_actor_geometric_occlusion_pipeline(geometric_observability=(pipeline_geometric('1'), pipeline_geometric('3')), multicamera_occlusion=pipeline_multicamera())

def test_empty_results_are_supported():
    source = pipeline_multicamera(())
    result = build_actor_geometric_occlusion_pipeline(geometric_observability=(), multicamera_occlusion=source)
    assert result.actor_count == 0
    assert result.combined_evidence.actor_evidence == ()

def test_pipeline_builds_combined_evidence_summary():
    result = build_actor_geometric_occlusion_pipeline(geometric_observability=(pipeline_geometric('2'), pipeline_geometric('1')), multicamera_occlusion=pipeline_multicamera())
    assert result.combined_summary.actor_count == 2
    assert dict(result.combined_summary.evidence_status_counts) == {'combined_evidence_available': 2}
    assert result.combined_summary.geometric_candidate_actor_count == 2
    assert result.combined_summary.geometric_candidate_with_sampled_surface_actor_count == 2
    assert result.combined_summary.geometric_candidate_with_winning_cells_actor_count == 2
    assert result.combined_summary.reasons == ()

def test_empty_pipeline_builds_empty_summary():
    result = build_actor_geometric_occlusion_pipeline(geometric_observability=(), multicamera_occlusion=pipeline_multicamera(()))
    assert result.combined_summary.actor_count == 0
    assert result.combined_summary.evidence_status_counts == ()

def context_geometric(*track_ids):
    actor_results = tuple((SimpleNamespace(track_id=track_id, projections=tuple((SimpleNamespace(camera_name=name) for name in ('front_wide', 'front_tele', 'cross_left', 'cross_right')))) for track_id in track_ids))
    return SimpleNamespace(actor_count=len(actor_results), actor_results=actor_results)

def context_combined(*track_ids):
    return tuple((SimpleNamespace(track_id=track_id) for track_id in track_ids))

def context_install_builder(monkeypatch, counts=None):
    calls = []
    counts = counts or {}

    def build(*, combined_evidence, projections_by_camera):
        calls.append((combined_evidence.track_id, tuple(projections_by_camera)))
        return tuple((SimpleNamespace(camera_name=f'camera_{index}') for index in range(counts.get(combined_evidence.track_id, 0))))
    monkeypatch.setattr(target, 'build_candidate_without_sampled_surface_projection_evidence', build)
    return calls

def test_joins_all_actors_in_track_order(monkeypatch):
    calls = context_install_builder(monkeypatch, {'2': 1})
    result = target.build_actor_geometric_occlusion_projection_context(geometric=context_geometric('2', '1'), combined_evidence=context_combined('1', '2'))
    assert result.actor_count == 2
    assert tuple((item.track_id for item in result.actor_context)) == ('1', '2')
    assert result.missing_surface_projection_context_count == 1
    assert tuple((item[0] for item in calls)) == ('1', '2')

def test_preserves_combined_evidence_objects(monkeypatch):
    context_install_builder(monkeypatch)
    source = context_combined('1')
    result = target.build_actor_geometric_occlusion_projection_context(geometric=context_geometric('1'), combined_evidence=source)
    assert result.actor_context[0].combined_evidence is source[0]

def test_context_actor_set_mismatch_is_rejected(monkeypatch):
    context_install_builder(monkeypatch)
    with pytest.raises(ValueError, match='Actor sets must match'):
        target.build_actor_geometric_occlusion_projection_context(geometric=context_geometric('1'), combined_evidence=context_combined('2'))

def test_geometric_count_mismatch_is_rejected(monkeypatch):
    context_install_builder(monkeypatch)
    value = context_geometric('1')
    value.actor_count = 2
    with pytest.raises(ValueError, match='are inconsistent'):
        target.build_actor_geometric_occlusion_projection_context(geometric=value, combined_evidence=context_combined('1'))

def test_context_empty_results_are_supported(monkeypatch):
    calls = context_install_builder(monkeypatch)
    result = target.build_actor_geometric_occlusion_projection_context(geometric=context_geometric(), combined_evidence=())
    assert result.actor_count == 0
    assert result.actor_context == ()
    assert result.missing_surface_projection_context_count == 0
    assert calls == []

def withcontext_install_stubs(monkeypatch, *, pipeline_count=2, context_count=2):
    calls = []
    evidence = tuple((SimpleNamespace(track_id=str(index)) for index in range(pipeline_count)))
    pipeline = SimpleNamespace(actor_count=pipeline_count, geometric=SimpleNamespace(source='geometric'), combined=SimpleNamespace(combined_evidence=SimpleNamespace(actor_evidence=evidence)))

    def build_pipeline(**kwargs):
        calls.append(('pipeline', kwargs))
        return pipeline
    context_items = tuple((SimpleNamespace(track_id=str(index)) for index in range(context_count)))

    def build_context(**kwargs):
        calls.append(('context', kwargs))
        return SimpleNamespace(actor_count=context_count, actor_context=context_items, missing_surface_projection_context_count=1)
    monkeypatch.setattr(target, 'build_actor_geometric_occlusion_from_geometry', build_pipeline)
    monkeypatch.setattr(target, 'build_actor_geometric_occlusion_projection_context', build_context)
    return (calls, pipeline)

def withcontext_arguments(actor_count=2):
    return {'actors': tuple(({'track_id': str(index)} for index in range(actor_count))), 'recorded_ego_message': {'ego': True}, 'cameras': {'camera': object()}, 'raster_width': 480, 'raster_height': 270, 'maximum_depth': 8, 'maximum_boundary_extent_px': 4.0}

def test_runs_pipeline_then_attaches_projection_context(monkeypatch):
    calls, pipeline = withcontext_install_stubs(monkeypatch)
    result = target.build_actor_geometric_occlusion_with_projection_context(**withcontext_arguments())
    assert tuple((name for name, _ in calls)) == ('pipeline', 'context')
    assert result.actor_count == 2
    assert result.pipeline is pipeline
    assert result.projection_context.missing_surface_projection_context_count == 1
    assert calls[1][1]['geometric'] is pipeline.geometric
    assert calls[1][1]['combined_evidence'] is pipeline.combined.combined_evidence.actor_evidence

def test_projection_and_occlusion_parameters_are_forwarded(monkeypatch):
    calls, _ = withcontext_install_stubs(monkeypatch)
    values = withcontext_arguments()
    values.update({'samples_per_edge': 9, 'near_plane_m': 0.01, 'maximum_chord_error_px': 0.5, 'maximum_adaptive_depth': 11, 'depth_tolerance_m': 1e-08})
    target.build_actor_geometric_occlusion_with_projection_context(**values)
    pipeline_call = calls[0][1]
    assert pipeline_call['samples_per_edge'] == 9
    assert pipeline_call['near_plane_m'] == 0.01
    assert pipeline_call['maximum_chord_error_px'] == 0.5
    assert pipeline_call['maximum_adaptive_depth'] == 11
    assert pipeline_call['depth_tolerance_m'] == 1e-08

def test_pipeline_count_mismatch_is_rejected(monkeypatch):
    withcontext_install_stubs(monkeypatch, pipeline_count=1, context_count=1)
    with pytest.raises(RuntimeError, match='differs from input'):
        target.build_actor_geometric_occlusion_with_projection_context(**withcontext_arguments(actor_count=2))

def test_context_count_mismatch_is_rejected(monkeypatch):
    withcontext_install_stubs(monkeypatch, pipeline_count=2, context_count=1)
    with pytest.raises(RuntimeError, match='Actor count differs'):
        target.build_actor_geometric_occlusion_with_projection_context(**withcontext_arguments())

def test_empty_input_is_supported(monkeypatch):
    calls, _ = withcontext_install_stubs(monkeypatch, pipeline_count=0, context_count=0)
    result = target.build_actor_geometric_occlusion_with_projection_context(**withcontext_arguments(actor_count=0))
    assert result.actor_count == 0
    assert result.projection_context.actor_context == ()
    assert tuple((name for name, _ in calls)) == ('pipeline', 'context')

def multisummary_evidence(camera_name, *, occupied=10, winning=5, occluded=5, occluders=('9',), evaluated=True, reasons=()):
    return build_camera_actor_occlusion_evidence(camera_name=camera_name, track_id='13', raster_width=480, raster_height=270, occupied_cell_count=occupied, winning_cell_count=winning, occluded_cell_count=occluded, occluding_actor_ids=occluders, actor_to_actor_occlusion_evaluated=evaluated, static_occlusion_evaluated=False, reasons=reasons)

def multisummary_complete_source():
    return (multisummary_evidence(CAMERA_NAMES[0], occupied=10, winning=10, occluded=0, occluders=()), multisummary_evidence(CAMERA_NAMES[1], occupied=10, winning=0, occluded=10, occluders=('29',)), multisummary_evidence(CAMERA_NAMES[2], occupied=10, winning=4, occluded=6, occluders=('18',)), multisummary_evidence(CAMERA_NAMES[3], occupied=0, winning=0, occluded=0, occluders=()))

def test_summarizes_threshold_free_cross_camera_features():
    result = summarize_actor_multicamera_occlusion(track_id='13', camera_evidence=multisummary_complete_source())
    assert result.evaluated_camera_names == tuple(CAMERA_NAMES[:3])
    assert result.no_sampled_surface_camera_names == (CAMERA_NAMES[3],)
    assert result.winning_camera_names == (CAMERA_NAMES[0], CAMERA_NAMES[2])
    assert result.fully_occluded_camera_names == (CAMERA_NAMES[1],)
    assert result.occluded_camera_names == (CAMERA_NAMES[1], CAMERA_NAMES[2])
    assert result.evaluated_camera_count == 3
    assert result.winning_camera_count == 2
    assert result.total_occupied_cell_count == 30
    assert result.total_winning_cell_count == 14
    assert result.total_occluded_cell_count == 16
    assert result.maximum_visible_fraction == 1.0
    assert result.occluding_actor_ids == ('18', '29')
    assert result.actor_to_actor_occlusion_evaluated
    assert not result.static_occlusion_evaluated
    assert result.reasons == ()

def test_all_no_surface_has_undefined_maximum_fraction():
    source = tuple((multisummary_evidence(camera_name, occupied=0, winning=0, occluded=0, occluders=()) for camera_name in CAMERA_NAMES))
    result = summarize_actor_multicamera_occlusion(track_id='13', camera_evidence=source)
    assert result.evaluated_camera_count == 0
    assert result.winning_camera_count == 0
    assert result.maximum_visible_fraction is None
    assert result.actor_to_actor_occlusion_evaluated
    assert result.reasons == ('no_camera_has_sampled_surface',)

def test_not_evaluated_camera_is_preserved_as_reason():
    source = list(multisummary_complete_source())
    source[3] = multisummary_evidence(CAMERA_NAMES[3], occupied=0, winning=0, occluded=0, occluders=(), evaluated=False, reasons=('camera_input_unavailable',))
    result = summarize_actor_multicamera_occlusion(track_id='13', camera_evidence=tuple(source))
    assert not result.actor_to_actor_occlusion_evaluated
    assert result.reasons == ('one_or_more_cameras_not_evaluated',)

def test_camera_order_is_required():
    source = multisummary_complete_source()
    with pytest.raises(ValueError, match='must follow CAMERA_NAMES'):
        summarize_actor_multicamera_occlusion(track_id='13', camera_evidence=tuple(reversed(source)))

def test_track_id_mismatch_is_rejected():
    source = list(multisummary_complete_source())
    wrong = build_camera_actor_occlusion_evidence(camera_name=CAMERA_NAMES[0], track_id='wrong', raster_width=480, raster_height=270, occupied_cell_count=10, winning_cell_count=10, occluded_cell_count=0, occluding_actor_ids=(), actor_to_actor_occlusion_evaluated=True)
    source[0] = wrong
    with pytest.raises(ValueError, match='must match'):
        summarize_actor_multicamera_occlusion(track_id='13', camera_evidence=tuple(source))

def test_multisummary_to_dict_uses_json_ready_lists():
    result = summarize_actor_multicamera_occlusion(track_id='13', camera_evidence=multisummary_complete_source())
    value = result.to_dict()
    assert isinstance(value['evaluated_camera_names'], list)
    assert isinstance(value['winning_camera_names'], list)
    assert isinstance(value['occluding_actor_ids'], list)
    assert isinstance(value['reasons'], list)

@dataclass(frozen=True)
class Cell_3:
    column: int
    row: int

@dataclass(frozen=True)
class SurfaceSample_2:
    cell: Cell_3
    depth_m: float

@dataclass(frozen=True)
class SurfaceRaster_2:
    cell_depths: tuple[SurfaceSample_2, ...]
    occupied_cell_count: int

def adapter_raster(*cells):
    samples = tuple((SurfaceSample_2(Cell_3(column, row), depth) for column, row, depth in cells))
    return SurfaceRaster_2(samples, len(samples))

def adapter_shared_case():
    near = ActorDepthRasterInput('near', adapter_raster((0, 0, 5.0), (1, 0, 5.0)))
    far = ActorDepthRasterInput('far', adapter_raster((0, 0, 10.0), (1, 0, 10.0), (2, 0, 10.0)))
    winner_0 = ZBufferCellWinner(Cell_3(0, 0), 'near', 5.0, near.surface_raster.cell_depths[0])
    winner_1 = ZBufferCellWinner(Cell_3(1, 0), 'near', 5.0, near.surface_raster.cell_depths[1])
    winner_2 = ZBufferCellWinner(Cell_3(2, 0), 'far', 10.0, far.surface_raster.cell_depths[2])
    zbuffer = ActorDepthZBuffer(cell_winners=(winner_0, winner_1, winner_2), actor_summaries=(ActorZBufferSummary('far', 3, 1, 2), ActorZBufferSummary('near', 2, 2, 0)), occupied_union_cell_count=3, contested_cell_count=2)
    return ((near, far), zbuffer)

def adapter_call(inputs=None, zbuffer=None):
    default_inputs, default_zbuffer = adapter_shared_case()
    return build_camera_occlusion_evidence_from_zbuffer(camera_name='cross_left', raster_width=480, raster_height=270, actor_rasters=default_inputs if inputs is None else inputs, zbuffer=default_zbuffer if zbuffer is None else zbuffer)

def test_extracts_occluder_ids_and_counts():
    result = {item.track_id: item for item in adapter_call()}
    assert result['far'].occupied_cell_count == 3
    assert result['far'].winning_cell_count == 1
    assert result['far'].occluded_cell_count == 2
    assert result['far'].visible_fraction == pytest.approx(1 / 3)
    assert result['far'].occluding_actor_ids == ('near',)
    assert result['near'].occluding_actor_ids == ()

def test_output_is_sorted_by_track_id():
    assert tuple((item.track_id for item in adapter_call())) == ('far', 'near')

def test_zero_surface_actor_is_preserved():
    empty = ActorDepthRasterInput('empty', adapter_raster())
    zbuffer = ActorDepthZBuffer(cell_winners=(), actor_summaries=(ActorZBufferSummary('empty', 0, 0, 0),), occupied_union_cell_count=0, contested_cell_count=0)
    result = adapter_call(inputs=(empty,), zbuffer=zbuffer)
    assert result[0].evidence_status == 'no_sampled_surface'
    assert result[0].visible_fraction is None

def test_summary_actor_set_must_match_inputs():
    inputs, zbuffer = adapter_shared_case()
    bad = ActorDepthZBuffer(cell_winners=zbuffer.cell_winners, actor_summaries=zbuffer.actor_summaries[:1], occupied_union_cell_count=zbuffer.occupied_union_cell_count, contested_cell_count=zbuffer.contested_cell_count)
    with pytest.raises(ValueError, match='must match'):
        adapter_call(inputs=inputs, zbuffer=bad)

def test_missing_winner_for_target_cell_is_rejected():
    inputs, zbuffer = adapter_shared_case()
    bad = ActorDepthZBuffer(cell_winners=zbuffer.cell_winners[:2], actor_summaries=zbuffer.actor_summaries, occupied_union_cell_count=2, contested_cell_count=zbuffer.contested_cell_count)
    with pytest.raises(ValueError, match='has no Z-buffer winner'):
        adapter_call(inputs=inputs, zbuffer=bad)

def test_duplicate_input_actor_ids_are_rejected():
    inputs, zbuffer = adapter_shared_case()
    with pytest.raises(ValueError, match='must be unique'):
        adapter_call(inputs=(inputs[0], inputs[0]), zbuffer=zbuffer)

def test_inconsistent_surface_count_is_rejected():
    inputs, zbuffer = adapter_shared_case()
    broken_surface = SurfaceRaster_2(inputs[0].surface_raster.cell_depths, 99)
    broken = ActorDepthRasterInput('near', broken_surface)
    with pytest.raises(ValueError, match='surface raster occupied count'):
        adapter_call(inputs=(broken, inputs[1]), zbuffer=zbuffer)

def test_unknown_winner_actor_is_rejected():
    inputs, zbuffer = adapter_shared_case()
    bad_first = ZBufferCellWinner(zbuffer.cell_winners[0].cell, 'unknown', zbuffer.cell_winners[0].depth_m, zbuffer.cell_winners[0].source_surface)
    bad = ActorDepthZBuffer(cell_winners=(bad_first, *zbuffer.cell_winners[1:]), actor_summaries=zbuffer.actor_summaries, occupied_union_cell_count=zbuffer.occupied_union_cell_count, contested_cell_count=zbuffer.contested_cell_count)
    with pytest.raises(ValueError, match='unknown Actor'):
        adapter_call(inputs=inputs, zbuffer=bad)

def cameraevidence_build(**overrides):
    values = {'camera_name': 'cross_left', 'track_id': '226', 'raster_width': 480, 'raster_height': 270, 'occupied_cell_count': 60, 'winning_cell_count': 0, 'occluded_cell_count': 60, 'occluding_actor_ids': ('80', '17', '132', '18'), 'actor_to_actor_occlusion_evaluated': True, 'static_occlusion_evaluated': False, 'reasons': ()}
    values.update(overrides)
    return build_camera_actor_occlusion_evidence(**values)

def test_evaluated_evidence_calculates_fraction():
    result = cameraevidence_build(occupied_cell_count=100, winning_cell_count=25, occluded_cell_count=75)
    assert result.visible_fraction == 0.25
    assert result.evidence_status == 'evaluated'
    assert result.actor_to_actor_occlusion_evaluated
    assert not result.static_occlusion_evaluated

def test_fully_occluded_evidence_preserves_zero():
    result = cameraevidence_build()
    assert result.visible_fraction == 0.0
    assert result.evidence_status == 'evaluated'

def test_no_sampled_surface_has_undefined_fraction():
    result = cameraevidence_build(occupied_cell_count=0, winning_cell_count=0, occluded_cell_count=0, occluding_actor_ids=())
    assert result.visible_fraction is None
    assert result.evidence_status == 'no_sampled_surface'

def test_unevaluated_evidence_has_explicit_status():
    result = cameraevidence_build(occupied_cell_count=0, winning_cell_count=0, occluded_cell_count=0, occluding_actor_ids=(), actor_to_actor_occlusion_evaluated=False, reasons=('camera_input_unavailable',))
    assert result.visible_fraction is None
    assert result.evidence_status == 'not_evaluated'

def test_occluding_actor_ids_are_sorted():
    result = cameraevidence_build()
    assert result.occluding_actor_ids == ('132', '17', '18', '80')

def test_cameraevidence_to_dict_uses_json_ready_lists():
    result = cameraevidence_build()
    value = result.to_dict()
    assert value['occlusion_evidence_format_version'] == OCCLUSION_EVIDENCE_FORMAT_VERSION
    assert isinstance(value['occluding_actor_ids'], list)
    assert isinstance(value['reasons'], list)

def test_inconsistent_counts_are_rejected():
    with pytest.raises(ValueError, match='must equal occupied'):
        cameraevidence_build(occupied_cell_count=60, winning_cell_count=1, occluded_cell_count=60)

def test_negative_counts_are_rejected():
    with pytest.raises(ValueError, match='non-negative'):
        cameraevidence_build(occupied_cell_count=-1)

def test_unevaluated_counts_are_rejected():
    with pytest.raises(ValueError, match='cannot contain Z-buffer counts'):
        cameraevidence_build(actor_to_actor_occlusion_evaluated=False)

def test_self_occlusion_is_rejected():
    with pytest.raises(ValueError, match='cannot occlude itself'):
        cameraevidence_build(occluding_actor_ids=('226',))

def test_duplicate_occluders_are_rejected():
    with pytest.raises(ValueError, match='must be unique'):
        cameraevidence_build(occluding_actor_ids=('17', '17'))

@pytest.mark.parametrize('field,value', [('raster_width', 0), ('raster_height', -1)])
def test_invalid_raster_dimensions_are_rejected(field, value):
    with pytest.raises(ValueError, match='must be positive'):
        cameraevidence_build(**{field: value})

def nosurface_projection(camera_name, *, truncated=False, valid=True):
    return ActorCameraProjection(camera_name=camera_name, track_id='13', actor_class='automobile', corner_count=8, edge_count=12, edge_samples_per_edge=0, camera_sample_count=36, positive_depth_sample_count=36, within_fov_sample_count=36, inside_image_sample_count=1, projected_bbox=None, clipped_bbox=None, projected_area_px=100.0, inside_image_area_px=2.0, inside_image_ratio=0.02, projected_hull=(), clipped_hull=(), projected_hull_area_px=90.0, inside_image_hull_area_px=1.5, inside_image_hull_ratio=0.016, projected_height_px=18.0, minimum_depth_m=20.0, maximum_depth_m=24.0, truncated=truncated, projection_valid=valid, failure_reason=None if valid else 'projected_box_outside_image')

def nosurface_projections(**overrides):
    values = {name: nosurface_projection(name) for name in CAMERA_NAMES}
    values.update(overrides)
    return values

def nosurface_evidence():
    return ActorGeometricOcclusionEvidence(track_id='13', actor_class='automobile', geometric_observability_status='candidate_visible', geometric_candidate_camera_names=('cross_left',), occlusion_evaluated_camera_names=(), occlusion_winning_camera_names=(), geometric_candidate_with_sampled_surface_camera_names=(), geometric_candidate_with_winning_cells_camera_names=(), geometric_candidate_without_sampled_surface_camera_names=('cross_left',), geometric_candidate_fully_occluded_camera_names=(), actor_to_actor_occlusion_evaluated=True, static_occlusion_evaluated=False, maximum_visible_fraction=None, total_occupied_cell_count=0, total_winning_cell_count=0, total_occluded_cell_count=0, occluding_actor_ids=(), evidence_status='candidate_without_sampled_surface', reasons=('no_geometric_candidate_has_sampled_surface',))

def test_reports_truncated_projection_context_without_causal_claim():
    result = build_candidate_without_sampled_surface_projection_evidence(combined_evidence=nosurface_evidence(), projections_by_camera=nosurface_projections(cross_left=nosurface_projection('cross_left', truncated=True)))
    assert len(result) == 1
    assert result[0].camera_name == 'cross_left'
    assert result[0].evidence_status == 'missing_surface_with_truncated_projection'
    assert result[0].reasons == ('geometric_candidate_without_sampled_surface', 'projection_truncated_by_image')
    assert result[0].inside_image_hull_area_px == 1.5

def test_reports_untruncated_projection_separately():
    result = build_candidate_without_sampled_surface_projection_evidence(combined_evidence=nosurface_evidence(), projections_by_camera=nosurface_projections())
    assert result[0].evidence_status == 'missing_surface_with_untruncated_projection'
    assert result[0].reasons == ('geometric_candidate_without_sampled_surface',)

def test_invalid_projection_has_explicit_status():
    result = build_candidate_without_sampled_surface_projection_evidence(combined_evidence=nosurface_evidence(), projections_by_camera=nosurface_projections(cross_left=nosurface_projection('cross_left', valid=False)))
    assert result[0].evidence_status == 'missing_surface_with_invalid_projection'
    assert result[0].reasons[-1] == 'projection_invalid'

def test_non_missing_candidate_cameras_are_not_emitted():
    result = build_candidate_without_sampled_surface_projection_evidence(combined_evidence=replace(nosurface_evidence(), geometric_candidate_camera_names=('front_wide', 'cross_left'), geometric_candidate_with_sampled_surface_camera_names=('front_wide',)), projections_by_camera=nosurface_projections())
    assert tuple((item.camera_name for item in result)) == ('cross_left',)

def test_projection_track_mismatch_is_rejected():
    values = nosurface_projections()
    values['cross_left'] = replace(values['cross_left'], track_id='other')
    with pytest.raises(ValueError, match='track_id differ'):
        build_candidate_without_sampled_surface_projection_evidence(combined_evidence=nosurface_evidence(), projections_by_camera=values)

def test_missing_camera_input_is_rejected():
    values = nosurface_projections()
    del values[CAMERA_NAMES[0]]
    with pytest.raises(ValueError, match='missing'):
        build_candidate_without_sampled_surface_projection_evidence(combined_evidence=nosurface_evidence(), projections_by_camera=values)

def test_to_dict_uses_json_ready_reasons():
    result = build_candidate_without_sampled_surface_projection_evidence(combined_evidence=nosurface_evidence(), projections_by_camera=nosurface_projections())
    assert isinstance(result[0].to_dict()['reasons'], list)

def multicamera_actor(track_id):
    return {'track_id': track_id}

def multicamera_cameras():
    return {camera_name: target.CameraOcclusionBuildInput(calibration=object(), image_width_px=1920, image_height_px=1080) for camera_name in CAMERA_NAMES}

def multicamera_install_stub(monkeypatch, *, reverse_one_camera=False):

    def build_one(*, actors, camera_name, raster_width, raster_height, **kwargs):
        track_ids = sorted((str(multicamera_actor['track_id']) for multicamera_actor in actors))
        if reverse_one_camera and camera_name == CAMERA_NAMES[0]:
            track_ids.reverse()
        evidence = tuple((build_camera_actor_occlusion_evidence(camera_name=camera_name, track_id=track_id, raster_width=raster_width, raster_height=raster_height, occupied_cell_count=10, winning_cell_count=5, occluded_cell_count=5, occluding_actor_ids=('other',), actor_to_actor_occlusion_evaluated=True, static_occlusion_evaluated=False) for track_id in track_ids))
        return SimpleNamespace(camera_name=camera_name, actor_count=len(actors), occlusion=SimpleNamespace(actor_evidence=evidence))
    monkeypatch.setattr(target, 'build_actor_camera_occlusion_from_geometry', build_one)

def multicamera_call(monkeypatch, **overrides):
    multicamera_install_stub(monkeypatch)
    values = {'actors': (multicamera_actor('2'), multicamera_actor('1')), 'recorded_ego_message': {'ego': True}, 'cameras': multicamera_cameras(), 'raster_width': 480, 'raster_height': 270, 'maximum_depth': 8, 'maximum_boundary_extent_px': 4.0}
    values.update(overrides)
    return target.build_multicamera_actor_occlusion_from_geometry(**values)

def test_builds_all_cameras_in_canonical_order(monkeypatch):
    result = multicamera_call(monkeypatch)
    assert tuple((item.camera_name for item in result.camera_results)) == tuple(CAMERA_NAMES)
    assert result.actor_count == 2

def test_groups_evidence_by_actor_and_camera(monkeypatch):
    result = multicamera_call(monkeypatch)
    assert tuple((item.track_id for item in result.actor_evidence)) == ('1', '2')
    for item in result.actor_evidence:
        assert tuple((evidence.camera_name for evidence in item.camera_evidence)) == tuple(CAMERA_NAMES)
        assert len(item.camera_evidence) == len(CAMERA_NAMES)

def test_multicamera_missing_camera_is_rejected(monkeypatch):
    source = multicamera_cameras()
    del source[CAMERA_NAMES[0]]
    with pytest.raises(ValueError, match='missing'):
        multicamera_call(monkeypatch, cameras=source)

def test_multicamera_extra_camera_is_rejected(monkeypatch):
    source = multicamera_cameras()
    source['extra'] = target.CameraOcclusionBuildInput(calibration=object(), image_width_px=1, image_height_px=1)
    with pytest.raises(ValueError, match='extra'):
        multicamera_call(monkeypatch, cameras=source)

def test_multicamera_duplicate_track_ids_are_rejected(monkeypatch):
    with pytest.raises(ValueError, match='must be unique'):
        multicamera_call(monkeypatch, actors=(multicamera_actor('1'), multicamera_actor('1')))

def test_camera_actor_set_mismatch_is_rejected(monkeypatch):
    multicamera_install_stub(monkeypatch, reverse_one_camera=True)
    with pytest.raises(RuntimeError, match='Actor IDs differ'):
        target.build_multicamera_actor_occlusion_from_geometry(actors=(multicamera_actor('2'), multicamera_actor('1')), recorded_ego_message={'ego': True}, cameras=multicamera_cameras(), raster_width=480, raster_height=270, maximum_depth=8, maximum_boundary_extent_px=4.0)

def test_builds_threshold_free_actor_summaries(monkeypatch):
    result = multicamera_call(monkeypatch)
    assert tuple((item.track_id for item in result.actor_summaries)) == ('1', '2')
    for item in result.actor_summaries:
        assert item.evaluated_camera_count == len(CAMERA_NAMES)
        assert item.winning_camera_count == len(CAMERA_NAMES)
        assert item.total_occupied_cell_count == 10 * len(CAMERA_NAMES)
        assert item.total_winning_cell_count == 5 * len(CAMERA_NAMES)
        assert item.total_occluded_cell_count == 5 * len(CAMERA_NAMES)
        assert item.maximum_visible_fraction == 0.5
        assert item.occluding_actor_ids == ('other',)
        assert item.actor_to_actor_occlusion_evaluated
        assert not item.static_occlusion_evaluated
        assert item.reasons == ()

def test_summary_camera_order_matches_grouped_evidence(monkeypatch):
    result = multicamera_call(monkeypatch)
    grouped_by_id = {item.track_id: item for item in result.actor_evidence}
    for summary in result.actor_summaries:
        grouped = grouped_by_id[summary.track_id]
        assert summary.evaluated_camera_names == tuple((item.camera_name for item in grouped.camera_evidence))
