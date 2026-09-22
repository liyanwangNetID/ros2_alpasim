"""Step 7 camera, multicamera, geometric-occlusion evidence, and adapter domain."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
from step7.projection import actor_box_corners_in_rig
from step7.raster import build_actor_camera_surface_depth_raster
from step7.raster import ActorDepthRasterInput
from step7.geometry import prepare_camera_facing_box_triangles
from typing import Sequence
from step7.raster import ActorDepthRasterInput, ActorDepthZBuffer, resolve_actor_depth_zbuffer
from step7.observability import ActorObservability
from collections import Counter
from dataclasses import asdict, dataclass
from step7.scene_facts import CAMERA_NAMES
from typing import Any, Mapping, Sequence
from step7.observability import MulticameraActorGeometricObservabilityResult, build_multicamera_actor_geometric_observability
from step7.observability import MulticameraActorGeometricObservabilityResult
from collections import defaultdict
from step7.raster import ActorDepthRasterInput, ActorDepthZBuffer
import math
from typing import Mapping
from step7.projection import ActorCameraProjection

@dataclass(frozen=True, slots=True)
class ActorCameraSurfaceDiagnostic:
    track_id: str
    prepared_source_triangle_count: int
    generated_triangle_sample_count: int
    occupied_cell_count: int
    zero_center_sample_triangle_count: int
    unresolved_boundary_count: int

@dataclass(frozen=True, slots=True)
class ActorCameraOcclusionFromGeometryResult:
    camera_name: str
    raster_width: int
    raster_height: int
    actor_count: int
    surface_diagnostics: tuple[ActorCameraSurfaceDiagnostic, ...]
    occlusion: CameraActorOcclusionPipelineResult

def _cameraresult_actor_id(actor: dict[str, Any]) -> str:
    value = actor.get('track_id')
    if value is None or str(value) == '':
        raise ValueError('Actor is missing a usable track_id')
    return str(value)

def build_actor_camera_occlusion_from_geometry(*, actors: Sequence[dict[str, Any]], recorded_ego_message: dict[str, Any], calibration: Any, camera_name: str, image_width_px: int, image_height_px: int, raster_width: int, raster_height: int, maximum_depth: int, maximum_boundary_extent_px: float, near_plane_m: float=0.001, depth_tolerance_m: float=1e-09) -> ActorCameraOcclusionFromGeometryResult:
    """Build all current-Actor camera rasters and shared occlusion evidence."""
    source = tuple(actors)
    track_ids = tuple((_cameraresult_actor_id(actor) for actor in source))
    if len(set(track_ids)) != len(track_ids):
        raise ValueError('Actor track_id values must be unique')
    if getattr(calibration, 'max_angle_rad', None) is None:
        raise ValueError('calibration.max_angle_rad is required')
    inputs: list[ActorDepthRasterInput] = []
    diagnostics: list[ActorCameraSurfaceDiagnostic] = []
    for actor in sorted(source, key=_cameraresult_actor_id):
        track_id = _cameraresult_actor_id(actor)
        corners_rig = actor_box_corners_in_rig(actor, recorded_ego_message=recorded_ego_message)
        corners_camera = tuple((calibration.rig_point_to_camera(point) for point in corners_rig))
        prepared = prepare_camera_facing_box_triangles(corners_camera, near_plane_m=near_plane_m)
        result = build_actor_camera_surface_depth_raster(tuple((item.vertices_camera for item in prepared)), calibration, max_angle_rad=calibration.max_angle_rad, maximum_depth=maximum_depth, maximum_boundary_extent_px=maximum_boundary_extent_px, image_width_px=image_width_px, image_height_px=image_height_px, raster_width=raster_width, raster_height=raster_height, near_plane_m=near_plane_m, depth_tolerance_m=depth_tolerance_m)
        inputs.append(ActorDepthRasterInput(track_id, result.surface_raster))
        diagnostics.append(ActorCameraSurfaceDiagnostic(track_id=track_id, prepared_source_triangle_count=len(prepared), generated_triangle_sample_count=result.generated_triangle_sample_count, occupied_cell_count=result.surface_raster.occupied_cell_count, zero_center_sample_triangle_count=result.zero_center_sample_triangle_count, unresolved_boundary_count=result.unresolved_boundary_count))
    occlusion = build_camera_actor_occlusion_pipeline(camera_name=camera_name, raster_width=raster_width, raster_height=raster_height, actor_rasters=tuple(inputs), depth_tolerance_m=depth_tolerance_m, static_occlusion_evaluated=False)
    evidence_ids = tuple((item.track_id for item in occlusion.actor_evidence))
    diagnostic_ids = tuple((item.track_id for item in diagnostics))
    if evidence_ids != diagnostic_ids:
        raise RuntimeError('surface diagnostics and occlusion evidence Actor IDs differ')
    return ActorCameraOcclusionFromGeometryResult(camera_name=camera_name, raster_width=raster_width, raster_height=raster_height, actor_count=len(source), surface_diagnostics=tuple(diagnostics), occlusion=occlusion)

@dataclass(frozen=True, slots=True)
class CameraActorOcclusionPipelineResult:
    camera_name: str
    raster_width: int
    raster_height: int
    zbuffer: ActorDepthZBuffer
    actor_evidence: tuple[CameraActorOcclusionEvidence, ...]

def build_camera_actor_occlusion_pipeline(*, camera_name: str, raster_width: int, raster_height: int, actor_rasters: Sequence[ActorDepthRasterInput], depth_tolerance_m: float=1e-09, static_occlusion_evaluated: bool=False) -> CameraActorOcclusionPipelineResult:
    """Resolve shared depth competition and produce evidence for every Actor."""
    inputs = tuple(actor_rasters)
    zbuffer = resolve_actor_depth_zbuffer(inputs, depth_tolerance_m=depth_tolerance_m)
    evidence = build_camera_occlusion_evidence_from_zbuffer(camera_name=camera_name, raster_width=raster_width, raster_height=raster_height, actor_rasters=inputs, zbuffer=zbuffer, static_occlusion_evaluated=static_occlusion_evaluated)
    input_ids = tuple(sorted((item.actor_id for item in inputs)))
    evidence_ids = tuple((item.track_id for item in evidence))
    summary_ids = tuple((item.actor_id for item in zbuffer.actor_summaries))
    if evidence_ids != input_ids:
        raise RuntimeError('occlusion evidence Actor IDs do not match inputs')
    if summary_ids != input_ids:
        raise RuntimeError('Z-buffer summary Actor IDs do not match inputs')
    return CameraActorOcclusionPipelineResult(camera_name=camera_name, raster_width=raster_width, raster_height=raster_height, zbuffer=zbuffer, actor_evidence=evidence)

@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionEvidenceSet:
    actor_count: int
    actor_evidence: tuple[ActorGeometricOcclusionEvidence, ...]

def build_actor_geometric_occlusion_evidence_set(*, geometric_observability: Sequence[ActorObservability], occlusion_summaries: Sequence[ActorMulticameraOcclusionSummary]) -> ActorGeometricOcclusionEvidenceSet:
    """Join complete geometric and occlusion Actor sets by track_id."""
    geometric = tuple(geometric_observability)
    occlusion = tuple(occlusion_summaries)
    geometric_by_id = {item.track_id: item for item in geometric}
    occlusion_by_id = {item.track_id: item for item in occlusion}
    if len(geometric_by_id) != len(geometric):
        raise ValueError('geometric observability contains duplicate track_id values')
    if len(occlusion_by_id) != len(occlusion):
        raise ValueError('occlusion summaries contain duplicate track_id values')
    if set(geometric_by_id) != set(occlusion_by_id):
        missing_geometric = sorted(set(occlusion_by_id) - set(geometric_by_id))
        missing_occlusion = sorted(set(geometric_by_id) - set(occlusion_by_id))
        raise ValueError(f'geometric and occlusion Actor sets must match; missing_geometric={missing_geometric}, missing_occlusion={missing_occlusion}')
    evidence = tuple((build_actor_geometric_occlusion_evidence(geometric=geometric_by_id[track_id], occlusion=occlusion_by_id[track_id]) for track_id in sorted(geometric_by_id)))
    if tuple((item.track_id for item in evidence)) != tuple(sorted(geometric_by_id)):
        raise RuntimeError('combined geometric-occlusion evidence order is inconsistent')
    return ActorGeometricOcclusionEvidenceSet(actor_count=len(evidence), actor_evidence=evidence)

@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionEvidenceSummary:
    actor_count: int
    evidence_status_counts: tuple[tuple[str, int], ...]
    geometric_candidate_actor_count: int
    geometric_candidate_with_sampled_surface_actor_count: int
    geometric_candidate_with_winning_cells_actor_count: int
    geometric_candidate_without_sampled_surface_actor_count: int
    geometric_candidate_fully_occluded_actor_count: int
    actor_with_occluder_count: int
    actor_to_actor_occlusion_complete_count: int
    static_occlusion_evaluated_count: int
    geometric_candidate_without_sampled_surface_track_ids: tuple[str, ...]
    geometric_candidate_fully_occluded_track_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value['evidence_status_counts'] = {key: count for key, count in self.evidence_status_counts}
        for key in ('geometric_candidate_without_sampled_surface_track_ids', 'geometric_candidate_fully_occluded_track_ids', 'reasons'):
            value[key] = list(value[key])
        return value

def summarize_actor_geometric_occlusion_evidence(evidence: Sequence[ActorGeometricOcclusionEvidence]) -> ActorGeometricOcclusionEvidenceSummary:
    """Aggregate a deterministic set of combined evidence records."""
    source = tuple(evidence)
    track_ids = tuple((item.track_id for item in source))
    if any((not track_id for track_id in track_ids)):
        raise ValueError('track_id values must be non-empty')
    if len(set(track_ids)) != len(track_ids):
        raise ValueError('track_id values must be unique')
    status_counts = Counter((item.evidence_status for item in source))
    candidate = tuple((item for item in source if item.geometric_candidate_camera_names))
    candidate_with_surface = tuple((item for item in source if item.geometric_candidate_with_sampled_surface_camera_names))
    candidate_with_winners = tuple((item for item in source if item.geometric_candidate_with_winning_cells_camera_names))
    candidate_without_surface = tuple((item for item in source if item.geometric_candidate_without_sampled_surface_camera_names))
    candidate_fully_occluded = tuple((item for item in source if item.geometric_candidate_fully_occluded_camera_names))
    reasons = []
    if any((not item.actor_to_actor_occlusion_evaluated for item in source)):
        reasons.append('one_or_more_actors_have_incomplete_occlusion_evidence')
    if any((item.static_occlusion_evaluated for item in source)):
        reasons.append('one_or_more_actors_have_static_occlusion_evidence')
    return ActorGeometricOcclusionEvidenceSummary(actor_count=len(source), evidence_status_counts=tuple(sorted(status_counts.items())), geometric_candidate_actor_count=len(candidate), geometric_candidate_with_sampled_surface_actor_count=len(candidate_with_surface), geometric_candidate_with_winning_cells_actor_count=len(candidate_with_winners), geometric_candidate_without_sampled_surface_actor_count=len(candidate_without_surface), geometric_candidate_fully_occluded_actor_count=len(candidate_fully_occluded), actor_with_occluder_count=sum((bool(item.occluding_actor_ids) for item in source)), actor_to_actor_occlusion_complete_count=sum((item.actor_to_actor_occlusion_evaluated for item in source)), static_occlusion_evaluated_count=sum((item.static_occlusion_evaluated for item in source)), geometric_candidate_without_sampled_surface_track_ids=tuple(sorted((item.track_id for item in candidate_without_surface))), geometric_candidate_fully_occluded_track_ids=tuple(sorted((item.track_id for item in candidate_fully_occluded))), reasons=tuple(reasons))

@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionEvidence:
    track_id: str
    actor_class: str
    geometric_observability_status: str
    geometric_candidate_camera_names: tuple[str, ...]
    occlusion_evaluated_camera_names: tuple[str, ...]
    occlusion_winning_camera_names: tuple[str, ...]
    geometric_candidate_with_sampled_surface_camera_names: tuple[str, ...]
    geometric_candidate_with_winning_cells_camera_names: tuple[str, ...]
    geometric_candidate_without_sampled_surface_camera_names: tuple[str, ...]
    geometric_candidate_fully_occluded_camera_names: tuple[str, ...]
    actor_to_actor_occlusion_evaluated: bool
    static_occlusion_evaluated: bool
    maximum_visible_fraction: float | None
    total_occupied_cell_count: int
    total_winning_cell_count: int
    total_occluded_cell_count: int
    occluding_actor_ids: tuple[str, ...]
    evidence_status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in ('geometric_candidate_camera_names', 'occlusion_evaluated_camera_names', 'occlusion_winning_camera_names', 'geometric_candidate_with_sampled_surface_camera_names', 'geometric_candidate_with_winning_cells_camera_names', 'geometric_candidate_without_sampled_surface_camera_names', 'geometric_candidate_fully_occluded_camera_names', 'occluding_actor_ids', 'reasons'):
            value[key] = list(value[key])
        return value

def _evidence_ordered_intersection(first: tuple[str, ...], second: set[str]) -> tuple[str, ...]:
    return tuple((value for value in first if value in second))

def build_actor_geometric_occlusion_evidence(*, geometric: ActorObservability, occlusion: ActorMulticameraOcclusionSummary) -> ActorGeometricOcclusionEvidence:
    """Join geometric candidates with threshold-free occlusion diagnostics."""
    if geometric.track_id != occlusion.track_id:
        raise ValueError('geometric and occlusion track_id values must match')
    canonical = tuple(CAMERA_NAMES)
    geometric_candidates = tuple(geometric.visible_in_cameras)
    if any((name not in CAMERA_NAMES for name in geometric_candidates)):
        raise ValueError('geometric candidate camera is not canonical')
    if geometric_candidates != tuple((name for name in canonical if name in set(geometric_candidates))):
        raise ValueError('geometric candidate cameras must use canonical order')
    evaluated_set = set(occlusion.evaluated_camera_names)
    winning_set = set(occlusion.winning_camera_names)
    no_surface_set = set(occlusion.no_sampled_surface_camera_names)
    fully_occluded_set = set(occlusion.fully_occluded_camera_names)
    candidate_with_surface = _evidence_ordered_intersection(geometric_candidates, evaluated_set)
    candidate_with_winners = _evidence_ordered_intersection(geometric_candidates, winning_set)
    candidate_without_surface = _evidence_ordered_intersection(geometric_candidates, no_surface_set)
    candidate_fully_occluded = _evidence_ordered_intersection(geometric_candidates, fully_occluded_set)
    reasons = []
    if not geometric_candidates:
        reasons.append('no_geometric_candidate_camera')
    if not occlusion.actor_to_actor_occlusion_evaluated:
        reasons.append('actor_to_actor_occlusion_not_fully_evaluated')
    if geometric_candidates and (not candidate_with_surface):
        reasons.append('no_geometric_candidate_has_sampled_surface')
    if not occlusion.actor_to_actor_occlusion_evaluated:
        evidence_status = 'occlusion_incomplete'
    elif not geometric_candidates:
        evidence_status = 'no_geometric_candidate'
    elif not candidate_with_surface:
        evidence_status = 'candidate_without_sampled_surface'
    else:
        evidence_status = 'combined_evidence_available'
    return ActorGeometricOcclusionEvidence(track_id=geometric.track_id, actor_class=geometric.actor_class, geometric_observability_status=geometric.observability_status, geometric_candidate_camera_names=geometric_candidates, occlusion_evaluated_camera_names=occlusion.evaluated_camera_names, occlusion_winning_camera_names=occlusion.winning_camera_names, geometric_candidate_with_sampled_surface_camera_names=candidate_with_surface, geometric_candidate_with_winning_cells_camera_names=candidate_with_winners, geometric_candidate_without_sampled_surface_camera_names=candidate_without_surface, geometric_candidate_fully_occluded_camera_names=candidate_fully_occluded, actor_to_actor_occlusion_evaluated=occlusion.actor_to_actor_occlusion_evaluated, static_occlusion_evaluated=occlusion.static_occlusion_evaluated, maximum_visible_fraction=occlusion.maximum_visible_fraction, total_occupied_cell_count=occlusion.total_occupied_cell_count, total_winning_cell_count=occlusion.total_winning_cell_count, total_occluded_cell_count=occlusion.total_occluded_cell_count, occluding_actor_ids=occlusion.occluding_actor_ids, evidence_status=evidence_status, reasons=tuple(reasons))

@dataclass(frozen=True, slots=True)
class CameraGeometricOcclusionBuildInput:
    calibration: Any
    image_width_px: int
    image_height_px: int

@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionFromGeometryResult:
    actor_count: int
    geometric: MulticameraActorGeometricObservabilityResult
    occlusion: MulticameraActorOcclusionResult
    combined: ActorGeometricOcclusionPipelineResult

def build_actor_geometric_occlusion_from_geometry(*, actors: Sequence[Mapping[str, Any]], recorded_ego_message: Mapping[str, Any], cameras: Mapping[str, CameraGeometricOcclusionBuildInput], raster_width: int, raster_height: int, maximum_depth: int, maximum_boundary_extent_px: float, samples_per_edge: int | None=None, near_plane_m: float=0.001, maximum_chord_error_px: float=1.0, maximum_adaptive_depth: int=14, depth_tolerance_m: float=1e-09) -> ActorGeometricOcclusionFromGeometryResult:
    """Run and join geometric and Actor-occlusion evidence from one snapshot."""
    expected = set(CAMERA_NAMES)
    actual = set(cameras)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f'Expected exactly CAMERA_NAMES; missing={missing}, extra={extra}')
    source = tuple(actors)
    calibrations = {name: cameras[name].calibration for name in CAMERA_NAMES}
    occlusion_cameras = {name: CameraOcclusionBuildInput(calibration=cameras[name].calibration, image_width_px=cameras[name].image_width_px, image_height_px=cameras[name].image_height_px) for name in CAMERA_NAMES}
    geometric = build_multicamera_actor_geometric_observability(actors=source, recorded_ego_message=recorded_ego_message, calibrations=calibrations, samples_per_edge=samples_per_edge, near_plane_m=near_plane_m, maximum_chord_error_px=maximum_chord_error_px, maximum_adaptive_depth=maximum_adaptive_depth)
    occlusion = build_multicamera_actor_occlusion_from_geometry(actors=source, recorded_ego_message=recorded_ego_message, cameras=occlusion_cameras, raster_width=raster_width, raster_height=raster_height, maximum_depth=maximum_depth, maximum_boundary_extent_px=maximum_boundary_extent_px, near_plane_m=near_plane_m, depth_tolerance_m=depth_tolerance_m)
    combined = build_actor_geometric_occlusion_pipeline(geometric_observability=geometric.actor_observability, multicamera_occlusion=occlusion)
    counts = (geometric.actor_count, occlusion.actor_count, combined.actor_count)
    if counts != (len(source), len(source), len(source)):
        raise RuntimeError('geometric, occlusion, and combined Actor counts are inconsistent')
    geometric_ids = tuple((item.track_id for item in geometric.actor_observability))
    occlusion_ids = tuple((item.track_id for item in occlusion.actor_summaries))
    combined_ids = tuple((item.track_id for item in combined.combined_evidence.actor_evidence))
    if geometric_ids != occlusion_ids or geometric_ids != combined_ids:
        raise RuntimeError('geometric, occlusion, and combined Actor IDs are inconsistent')
    return ActorGeometricOcclusionFromGeometryResult(actor_count=len(source), geometric=geometric, occlusion=occlusion, combined=combined)

@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionPipelineResult:
    actor_count: int
    geometric_observability: tuple[ActorObservability, ...]
    multicamera_occlusion: MulticameraActorOcclusionResult
    combined_evidence: ActorGeometricOcclusionEvidenceSet
    combined_summary: ActorGeometricOcclusionEvidenceSummary

def build_actor_geometric_occlusion_pipeline(*, geometric_observability: Sequence[ActorObservability], multicamera_occlusion: MulticameraActorOcclusionResult) -> ActorGeometricOcclusionPipelineResult:
    """Join complete upstream geometric and multicamera occlusion results."""
    geometric = tuple(geometric_observability)
    if multicamera_occlusion.actor_count != len(multicamera_occlusion.actor_summaries):
        raise ValueError('multicamera occlusion actor_count and summaries are inconsistent')
    if multicamera_occlusion.actor_count != len(multicamera_occlusion.actor_evidence):
        raise ValueError('multicamera occlusion actor_count and evidence are inconsistent')
    combined = build_actor_geometric_occlusion_evidence_set(geometric_observability=geometric, occlusion_summaries=multicamera_occlusion.actor_summaries)
    if combined.actor_count != multicamera_occlusion.actor_count:
        raise ValueError('geometric and multicamera occlusion Actor counts must match')
    summary = summarize_actor_geometric_occlusion_evidence(combined.actor_evidence)
    if summary.actor_count != combined.actor_count:
        raise RuntimeError('combined evidence summary Actor count is inconsistent')
    geometric_ids = tuple(sorted((item.track_id for item in geometric)))
    combined_ids = tuple((item.track_id for item in combined.actor_evidence))
    summary_ids = tuple((item.track_id for item in multicamera_occlusion.actor_summaries))
    if combined_ids != geometric_ids or combined_ids != summary_ids:
        raise RuntimeError('pipeline geometric, occlusion, and combined Actor IDs differ')
    return ActorGeometricOcclusionPipelineResult(actor_count=combined.actor_count, geometric_observability=tuple(sorted(geometric, key=lambda item: item.track_id)), multicamera_occlusion=multicamera_occlusion, combined_evidence=combined, combined_summary=summary)

@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionProjectionContext:
    track_id: str
    combined_evidence: ActorGeometricOcclusionEvidence
    candidate_without_sampled_surface_projection_evidence: tuple[CandidateWithoutSampledSurfaceProjectionEvidence, ...]

@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionProjectionContextResult:
    actor_count: int
    actor_context: tuple[ActorGeometricOcclusionProjectionContext, ...]
    missing_surface_projection_context_count: int

def build_actor_geometric_occlusion_projection_context(*, geometric: MulticameraActorGeometricObservabilityResult, combined_evidence: tuple[ActorGeometricOcclusionEvidence, ...]) -> ActorGeometricOcclusionProjectionContextResult:
    """Join all Actor projections with combined evidence by track_id."""
    geometric_results = tuple(geometric.actor_results)
    geometric_by_id = {item.track_id: item for item in geometric_results}
    if len(geometric_by_id) != len(geometric_results):
        raise ValueError('geometric result contains duplicate track_id values')
    combined = tuple(combined_evidence)
    combined_by_id = {item.track_id: item for item in combined}
    if len(combined_by_id) != len(combined):
        raise ValueError('combined evidence contains duplicate track_id values')
    if geometric.actor_count != len(geometric_results):
        raise ValueError('geometric actor_count and actor_results are inconsistent')
    if set(geometric_by_id) != set(combined_by_id):
        missing_geometric = sorted(set(combined_by_id) - set(geometric_by_id))
        missing_combined = sorted(set(geometric_by_id) - set(combined_by_id))
        raise ValueError(f'geometric and combined Actor sets must match; missing_geometric={missing_geometric}, missing_combined={missing_combined}')
    contexts = []
    context_count = 0
    for track_id in sorted(combined_by_id):
        geometric_item = geometric_by_id[track_id]
        projections_by_camera = {projection.camera_name: projection for projection in geometric_item.projections}
        projection_context = build_candidate_without_sampled_surface_projection_evidence(combined_evidence=combined_by_id[track_id], projections_by_camera=projections_by_camera)
        context_count += len(projection_context)
        contexts.append(ActorGeometricOcclusionProjectionContext(track_id=track_id, combined_evidence=combined_by_id[track_id], candidate_without_sampled_surface_projection_evidence=projection_context))
    return ActorGeometricOcclusionProjectionContextResult(actor_count=len(contexts), actor_context=tuple(contexts), missing_surface_projection_context_count=context_count)

@dataclass(frozen=True, slots=True)
class ActorGeometricOcclusionWithProjectionContextResult:
    actor_count: int
    pipeline: ActorGeometricOcclusionFromGeometryResult
    projection_context: ActorGeometricOcclusionProjectionContextResult

def build_actor_geometric_occlusion_with_projection_context(*, actors: Sequence[Mapping[str, Any]], recorded_ego_message: Mapping[str, Any], cameras: Mapping[str, CameraGeometricOcclusionBuildInput], raster_width: int, raster_height: int, maximum_depth: int, maximum_boundary_extent_px: float, samples_per_edge: int | None=None, near_plane_m: float=0.001, maximum_chord_error_px: float=1.0, maximum_adaptive_depth: int=14, depth_tolerance_m: float=1e-09) -> ActorGeometricOcclusionWithProjectionContextResult:
    """Run the existing pipeline and attach threshold-free projection context."""
    source = tuple(actors)
    pipeline = build_actor_geometric_occlusion_from_geometry(actors=source, recorded_ego_message=recorded_ego_message, cameras=cameras, raster_width=raster_width, raster_height=raster_height, maximum_depth=maximum_depth, maximum_boundary_extent_px=maximum_boundary_extent_px, samples_per_edge=samples_per_edge, near_plane_m=near_plane_m, maximum_chord_error_px=maximum_chord_error_px, maximum_adaptive_depth=maximum_adaptive_depth, depth_tolerance_m=depth_tolerance_m)
    context = build_actor_geometric_occlusion_projection_context(geometric=pipeline.geometric, combined_evidence=pipeline.combined.combined_evidence.actor_evidence)
    if pipeline.actor_count != len(source):
        raise RuntimeError('pipeline Actor count differs from input')
    if context.actor_count != pipeline.actor_count:
        raise RuntimeError('projection-context Actor count differs from pipeline')
    pipeline_ids = tuple((item.track_id for item in pipeline.combined.combined_evidence.actor_evidence))
    context_ids = tuple((item.track_id for item in context.actor_context))
    if context_ids != pipeline_ids:
        raise RuntimeError('projection-context Actor IDs differ from combined evidence')
    return ActorGeometricOcclusionWithProjectionContextResult(actor_count=pipeline.actor_count, pipeline=pipeline, projection_context=context)

@dataclass(frozen=True, slots=True)
class ActorMulticameraOcclusionSummary:
    track_id: str
    evaluated_camera_names: tuple[str, ...]
    no_sampled_surface_camera_names: tuple[str, ...]
    winning_camera_names: tuple[str, ...]
    fully_occluded_camera_names: tuple[str, ...]
    occluded_camera_names: tuple[str, ...]
    evaluated_camera_count: int
    winning_camera_count: int
    total_occupied_cell_count: int
    total_winning_cell_count: int
    total_occluded_cell_count: int
    maximum_visible_fraction: float | None
    occluding_actor_ids: tuple[str, ...]
    actor_to_actor_occlusion_evaluated: bool
    static_occlusion_evaluated: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for key in ('evaluated_camera_names', 'no_sampled_surface_camera_names', 'winning_camera_names', 'fully_occluded_camera_names', 'occluded_camera_names', 'occluding_actor_ids', 'reasons'):
            value[key] = list(value[key])
        return value

def summarize_actor_multicamera_occlusion(*, track_id: str, camera_evidence: Sequence[CameraActorOcclusionEvidence]) -> ActorMulticameraOcclusionSummary:
    """Derive threshold-free cross-camera features for one Actor."""
    if not isinstance(track_id, str) or not track_id:
        raise ValueError('track_id must be a non-empty string')
    source = tuple(camera_evidence)
    expected_order = tuple(CAMERA_NAMES)
    actual_order = tuple((item.camera_name for item in source))
    if actual_order != expected_order:
        raise ValueError(f'camera evidence must follow CAMERA_NAMES; actual={actual_order}')
    if any((item.track_id != track_id for item in source)):
        raise ValueError('camera evidence track_id values must match track_id')
    if any((item.static_occlusion_evaluated for item in source)):
        raise ValueError('static occlusion must remain unevaluated in Step 7E v0.1')
    allowed_statuses = {'evaluated', 'no_sampled_surface', 'not_evaluated'}
    if any((item.evidence_status not in allowed_statuses for item in source)):
        raise ValueError('unexpected camera occlusion evidence status')
    evaluated = tuple((item for item in source if item.evidence_status == 'evaluated'))
    no_surface = tuple((item for item in source if item.evidence_status == 'no_sampled_surface'))
    not_evaluated = tuple((item for item in source if item.evidence_status == 'not_evaluated'))
    winning = tuple((item for item in evaluated if item.winning_cell_count > 0))
    fully_occluded = tuple((item for item in evaluated if item.winning_cell_count == 0))
    occluded = tuple((item for item in evaluated if item.occluded_cell_count > 0))
    reasons = []
    if not_evaluated:
        reasons.append('one_or_more_cameras_not_evaluated')
    if not evaluated:
        reasons.append('no_camera_has_sampled_surface')
    occluders = tuple(sorted({actor_id for item in evaluated for actor_id in item.occluding_actor_ids}))
    maximum_visible_fraction = max((item.visible_fraction for item in evaluated)) if evaluated else None
    return ActorMulticameraOcclusionSummary(track_id=track_id, evaluated_camera_names=tuple((item.camera_name for item in evaluated)), no_sampled_surface_camera_names=tuple((item.camera_name for item in no_surface)), winning_camera_names=tuple((item.camera_name for item in winning)), fully_occluded_camera_names=tuple((item.camera_name for item in fully_occluded)), occluded_camera_names=tuple((item.camera_name for item in occluded)), evaluated_camera_count=len(evaluated), winning_camera_count=len(winning), total_occupied_cell_count=sum((item.occupied_cell_count for item in evaluated)), total_winning_cell_count=sum((item.winning_cell_count for item in evaluated)), total_occluded_cell_count=sum((item.occluded_cell_count for item in evaluated)), maximum_visible_fraction=maximum_visible_fraction, occluding_actor_ids=occluders, actor_to_actor_occlusion_evaluated=len(not_evaluated) == 0, static_occlusion_evaluated=False, reasons=tuple(reasons))

def build_camera_occlusion_evidence_from_zbuffer(*, camera_name: str, raster_width: int, raster_height: int, actor_rasters: Sequence[ActorDepthRasterInput], zbuffer: ActorDepthZBuffer, static_occlusion_evaluated: bool=False) -> tuple[CameraActorOcclusionEvidence, ...]:
    """Build one threshold-free evidence record for every input Actor.

    Occluding Actor IDs are derived from target-occupied cells won by another
    Actor. The adapter validates that Z-buffer summaries and winners are
    consistent with the supplied Actor surface rasters.
    """
    inputs = tuple(actor_rasters)
    input_ids = tuple((item.actor_id for item in inputs))
    if any((not isinstance(actor_id, str) or not actor_id for actor_id in input_ids)):
        raise ValueError('actor_id must be a non-empty string')
    if len(set(input_ids)) != len(input_ids):
        raise ValueError('actor_id values must be unique')
    summaries = {item.actor_id: item for item in zbuffer.actor_summaries}
    if len(summaries) != len(zbuffer.actor_summaries):
        raise ValueError('Z-buffer summaries contain duplicate actor_id values')
    if set(summaries) != set(input_ids):
        raise ValueError('Z-buffer summary Actor IDs must match input Actor IDs')
    winners_by_cell = {item.cell: item for item in zbuffer.cell_winners}
    if len(winners_by_cell) != len(zbuffer.cell_winners):
        raise ValueError('Z-buffer contains duplicate winner cells')
    if len(winners_by_cell) != zbuffer.occupied_union_cell_count:
        raise ValueError('Z-buffer occupied union count is inconsistent')
    if any((item.actor_id not in summaries for item in zbuffer.cell_winners)):
        raise ValueError('Z-buffer winner references an unknown Actor')
    occluders_by_actor: dict[str, set[str]] = defaultdict(set)
    for actor_input in inputs:
        actor_id = actor_input.actor_id
        surface = actor_input.surface_raster
        if surface.occupied_cell_count != len(surface.cell_depths):
            raise ValueError('surface raster occupied count is inconsistent')
        seen_cells = set()
        for sample in surface.cell_depths:
            if sample.cell in seen_cells:
                raise ValueError('surface raster contains duplicate cells')
            seen_cells.add(sample.cell)
            winner = winners_by_cell.get(sample.cell)
            if winner is None:
                raise ValueError('target occupied cell has no Z-buffer winner')
            if winner.actor_id != actor_id:
                occluders_by_actor[actor_id].add(winner.actor_id)
        summary = summaries[actor_id]
        if summary.occupied_cell_count != surface.occupied_cell_count:
            raise ValueError('Z-buffer summary occupied count is inconsistent')
        if summary.winning_cell_count + summary.occluded_cell_count != summary.occupied_cell_count:
            raise ValueError('Z-buffer summary counts are inconsistent')
    evidence = []
    for actor_input in sorted(inputs, key=lambda item: item.actor_id):
        summary = summaries[actor_input.actor_id]
        evidence.append(build_camera_actor_occlusion_evidence(camera_name=camera_name, track_id=actor_input.actor_id, raster_width=raster_width, raster_height=raster_height, occupied_cell_count=summary.occupied_cell_count, winning_cell_count=summary.winning_cell_count, occluded_cell_count=summary.occluded_cell_count, occluding_actor_ids=tuple(occluders_by_actor[actor_input.actor_id]), actor_to_actor_occlusion_evaluated=True, static_occlusion_evaluated=static_occlusion_evaluated, reasons=()))
    return tuple(evidence)
OCCLUSION_EVIDENCE_FORMAT_VERSION = '0.1-draft'

@dataclass(frozen=True, slots=True)
class CameraActorOcclusionEvidence:
    occlusion_evidence_format_version: str
    camera_name: str
    track_id: str
    raster_width: int
    raster_height: int
    occupied_cell_count: int
    winning_cell_count: int
    occluded_cell_count: int
    visible_fraction: float | None
    occluding_actor_ids: tuple[str, ...]
    actor_to_actor_occlusion_evaluated: bool
    static_occlusion_evaluated: bool
    evidence_status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value['occluding_actor_ids'] = list(self.occluding_actor_ids)
        value['reasons'] = list(self.reasons)
        return value

def build_camera_actor_occlusion_evidence(*, camera_name: str, track_id: str, raster_width: int, raster_height: int, occupied_cell_count: int, winning_cell_count: int, occluded_cell_count: int, occluding_actor_ids: Sequence[str], actor_to_actor_occlusion_evaluated: bool, static_occlusion_evaluated: bool=False, reasons: Sequence[str]=()) -> CameraActorOcclusionEvidence:
    """Validate counts and construct threshold-free occlusion evidence."""
    if not isinstance(camera_name, str) or not camera_name:
        raise ValueError('camera_name must be a non-empty string')
    if not isinstance(track_id, str) or not track_id:
        raise ValueError('track_id must be a non-empty string')
    dimensions = (raster_width, raster_height)
    if any((isinstance(value, bool) or not isinstance(value, int) for value in dimensions)):
        raise TypeError('raster dimensions must be integers')
    if any((value <= 0 for value in dimensions)):
        raise ValueError('raster dimensions must be positive')
    counts = (occupied_cell_count, winning_cell_count, occluded_cell_count)
    if any((isinstance(value, bool) or not isinstance(value, int) for value in counts)):
        raise TypeError('occlusion counts must be integers')
    if any((value < 0 for value in counts)):
        raise ValueError('occlusion counts must be non-negative')
    if winning_cell_count + occluded_cell_count != occupied_cell_count:
        raise ValueError('winning and occluded counts must equal occupied count')
    occluders = tuple(occluding_actor_ids)
    if any((not isinstance(actor_id, str) or not actor_id for actor_id in occluders)):
        raise ValueError('occluding Actor IDs must be non-empty strings')
    if track_id in occluders:
        raise ValueError('an Actor cannot occlude itself')
    if len(set(occluders)) != len(occluders):
        raise ValueError('occluding Actor IDs must be unique')
    occluders = tuple(sorted(occluders))
    reason_values = tuple(reasons)
    if any((not isinstance(reason, str) or not reason for reason in reason_values)):
        raise ValueError('reasons must be non-empty strings')
    if actor_to_actor_occlusion_evaluated:
        if occupied_cell_count:
            visible_fraction = winning_cell_count / occupied_cell_count
            evidence_status = 'evaluated'
        else:
            visible_fraction = None
            evidence_status = 'no_sampled_surface'
    else:
        if any(counts):
            raise ValueError('unevaluated evidence cannot contain Z-buffer counts')
        if occluders:
            raise ValueError('unevaluated evidence cannot contain occluding Actor IDs')
        visible_fraction = None
        evidence_status = 'not_evaluated'
    if visible_fraction is not None and (not math.isfinite(visible_fraction) or not 0.0 <= visible_fraction <= 1.0):
        raise ValueError('visible_fraction must be finite and in [0, 1]')
    return CameraActorOcclusionEvidence(occlusion_evidence_format_version=OCCLUSION_EVIDENCE_FORMAT_VERSION, camera_name=camera_name, track_id=track_id, raster_width=raster_width, raster_height=raster_height, occupied_cell_count=occupied_cell_count, winning_cell_count=winning_cell_count, occluded_cell_count=occluded_cell_count, visible_fraction=visible_fraction, occluding_actor_ids=occluders, actor_to_actor_occlusion_evaluated=actor_to_actor_occlusion_evaluated, static_occlusion_evaluated=static_occlusion_evaluated, evidence_status=evidence_status, reasons=reason_values)

@dataclass(frozen=True, slots=True)
class CandidateWithoutSampledSurfaceProjectionEvidence:
    camera_name: str
    track_id: str
    actor_class: str
    projection_valid: bool
    projection_truncated: bool
    inside_image_hull_area_px: float
    projected_height_px: float
    inside_image_hull_ratio: float
    minimum_depth_m: float | None
    maximum_depth_m: float | None
    evidence_status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value['reasons'] = list(self.reasons)
        return value

def build_candidate_without_sampled_surface_projection_evidence(*, combined_evidence: ActorGeometricOcclusionEvidence, projections_by_camera: Mapping[str, ActorCameraProjection]) -> tuple[CandidateWithoutSampledSurfaceProjectionEvidence, ...]:
    """Build one projection-context record per missing-surface candidate camera."""
    expected = set(CAMERA_NAMES)
    actual = set(projections_by_camera)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f'Expected exactly CAMERA_NAMES; missing={missing}, extra={extra}')
    missing_surface_cameras = tuple(combined_evidence.geometric_candidate_without_sampled_surface_camera_names)
    candidate_set = set(combined_evidence.geometric_candidate_camera_names)
    if any((name not in candidate_set for name in missing_surface_cameras)):
        raise ValueError('missing-surface cameras must be geometric candidate cameras')
    records = []
    for camera_name in CAMERA_NAMES:
        if camera_name not in missing_surface_cameras:
            continue
        projection = projections_by_camera[camera_name]
        if projection.camera_name != camera_name:
            raise ValueError('projection camera_name differs from mapping key')
        if projection.track_id != combined_evidence.track_id:
            raise ValueError('projection and combined evidence track_id differ')
        if projection.actor_class != combined_evidence.actor_class:
            raise ValueError('projection and combined evidence actor_class differ')
        reasons = ['geometric_candidate_without_sampled_surface']
        if not projection.projection_valid:
            status = 'missing_surface_with_invalid_projection'
            reasons.append('projection_invalid')
        elif projection.truncated:
            status = 'missing_surface_with_truncated_projection'
            reasons.append('projection_truncated_by_image')
        else:
            status = 'missing_surface_with_untruncated_projection'
        records.append(CandidateWithoutSampledSurfaceProjectionEvidence(camera_name=camera_name, track_id=combined_evidence.track_id, actor_class=combined_evidence.actor_class, projection_valid=projection.projection_valid, projection_truncated=projection.truncated, inside_image_hull_area_px=float(projection.inside_image_hull_area_px), projected_height_px=float(projection.projected_height_px), inside_image_hull_ratio=float(projection.inside_image_hull_ratio), minimum_depth_m=projection.minimum_depth_m, maximum_depth_m=projection.maximum_depth_m, evidence_status=status, reasons=tuple(reasons)))
    return tuple(records)

@dataclass(frozen=True, slots=True)
class CameraOcclusionBuildInput:
    calibration: Any
    image_width_px: int
    image_height_px: int

@dataclass(frozen=True, slots=True)
class ActorMulticameraOcclusionEvidence:
    track_id: str
    camera_evidence: tuple[CameraActorOcclusionEvidence, ...]

@dataclass(frozen=True, slots=True)
class MulticameraActorOcclusionResult:
    actor_count: int
    camera_results: tuple[ActorCameraOcclusionFromGeometryResult, ...]
    actor_evidence: tuple[ActorMulticameraOcclusionEvidence, ...]
    actor_summaries: tuple[ActorMulticameraOcclusionSummary, ...]

def _multicamera_track_id(actor: dict[str, Any]) -> str:
    value = actor.get('track_id')
    if value is None or str(value) == '':
        raise ValueError('Actor is missing a usable track_id')
    return str(value)

def build_multicamera_actor_occlusion_from_geometry(*, actors: Sequence[dict[str, Any]], recorded_ego_message: dict[str, Any], cameras: Mapping[str, CameraOcclusionBuildInput], raster_width: int, raster_height: int, maximum_depth: int, maximum_boundary_extent_px: float, near_plane_m: float=0.001, depth_tolerance_m: float=1e-09) -> MulticameraActorOcclusionResult:
    """Build four independent camera Z-buffers and group evidence by Actor."""
    expected_cameras = tuple(CAMERA_NAMES)
    expected_set = set(expected_cameras)
    actual_set = set(cameras)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(f'Expected exactly CAMERA_NAMES; missing={missing}, extra={extra}')
    source = tuple(actors)
    track_ids = tuple((_multicamera_track_id(actor) for actor in source))
    if len(set(track_ids)) != len(track_ids):
        raise ValueError('Actor track_id values must be unique')
    ordered_track_ids = tuple(sorted(track_ids))
    camera_results = []
    evidence_by_actor: dict[str, list[CameraActorOcclusionEvidence]] = {track_id: [] for track_id in ordered_track_ids}
    for camera_name in expected_cameras:
        camera = cameras[camera_name]
        result = build_actor_camera_occlusion_from_geometry(actors=source, recorded_ego_message=recorded_ego_message, calibration=camera.calibration, camera_name=camera_name, image_width_px=camera.image_width_px, image_height_px=camera.image_height_px, raster_width=raster_width, raster_height=raster_height, maximum_depth=maximum_depth, maximum_boundary_extent_px=maximum_boundary_extent_px, near_plane_m=near_plane_m, depth_tolerance_m=depth_tolerance_m)
        camera_results.append(result)
        evidence_ids = tuple((item.track_id for item in result.occlusion.actor_evidence))
        if evidence_ids != ordered_track_ids:
            raise RuntimeError(f'Camera {camera_name} evidence Actor IDs differ from inputs')
        for item in result.occlusion.actor_evidence:
            if item.camera_name != camera_name:
                raise RuntimeError(f'Camera evidence mismatch: expected {camera_name}, got {item.camera_name}')
            evidence_by_actor[item.track_id].append(item)
    grouped = tuple((ActorMulticameraOcclusionEvidence(track_id=track_id, camera_evidence=tuple(evidence_by_actor[track_id])) for track_id in ordered_track_ids))
    for item in grouped:
        camera_order = tuple((evidence.camera_name for evidence in item.camera_evidence))
        if camera_order != expected_cameras:
            raise RuntimeError(f'Actor {item.track_id} camera evidence order is invalid')
    summaries = tuple((summarize_actor_multicamera_occlusion(track_id=item.track_id, camera_evidence=item.camera_evidence) for item in grouped))
    summary_ids = tuple((item.track_id for item in summaries))
    if summary_ids != ordered_track_ids:
        raise RuntimeError('multicamera occlusion summary Actor IDs differ from inputs')
    return MulticameraActorOcclusionResult(actor_count=len(source), camera_results=tuple(camera_results), actor_evidence=grouped, actor_summaries=summaries)
