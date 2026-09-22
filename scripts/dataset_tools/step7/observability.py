"""Step 7 geometric observability and frozen dynamic-occlusion policy domain.

Contains camera-level geometric decisions, multi-camera aggregation, frozen shadow policy, and deterministic Actor decisions."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any, Final, Mapping, Sequence
from typing import Any, Mapping, Sequence
from step7.projection import ActorCameraProjection, project_actor_box_to_camera
from step7.scene_facts import CAMERA_NAMES, OBSERVABILITY_FORMAT_VERSION
import math
FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT: Final = 'below_minimum_projected_height'
FAILURE_BELOW_PRIMARY_AREA_AND_HEIGHT: Final = 'below_primary_area_and_height'
FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO: Final = 'below_minimum_inside_image_hull_ratio'

@dataclass(frozen=True)
class GeometricObservabilityDecision:
    candidate: bool
    failure_reason: str | None

@dataclass(frozen=True)
class HeightPolicy:
    minimum_projected_height_px: float

@dataclass(frozen=True)
class TelePolicy:
    minimum_inside_image_hull_area_px: float
    minimum_projected_height_px: float
    minimum_inside_image_hull_ratio: float
HEIGHT_POLICIES: Final = {'front_wide': HeightPolicy(minimum_projected_height_px=5.0), 'cross_left': HeightPolicy(minimum_projected_height_px=6.0), 'cross_right': HeightPolicy(minimum_projected_height_px=6.0)}
FRONT_TELE_POLICY: Final = TelePolicy(minimum_inside_image_hull_area_px=512.0, minimum_projected_height_px=16.0, minimum_inside_image_hull_ratio=0.1)

def evaluate_geometric_observability(*, camera_name: str, inside_image_hull_area_px: float, projected_height_px: float, inside_image_hull_ratio: float) -> GeometricObservabilityDecision:
    """Evaluate one valid geometric projection.

    Inputs are expected to come from a valid ActorCameraProjection. Thresholds
    are inclusive, so a value exactly on a boundary passes that boundary.
    """
    if camera_name not in CAMERA_NAMES:
        raise ValueError(f'Unsupported camera_name: {camera_name!r}')
    if inside_image_hull_area_px < 0.0:
        raise ValueError('inside_image_hull_area_px must be non-negative')
    if projected_height_px < 0.0:
        raise ValueError('projected_height_px must be non-negative')
    if not 0.0 <= inside_image_hull_ratio <= 1.0:
        raise ValueError('inside_image_hull_ratio must be in [0, 1]')
    height_policy = HEIGHT_POLICIES.get(camera_name)
    if height_policy is not None:
        if projected_height_px < height_policy.minimum_projected_height_px:
            return GeometricObservabilityDecision(candidate=False, failure_reason=FAILURE_BELOW_MINIMUM_PROJECTED_HEIGHT)
        return GeometricObservabilityDecision(candidate=True, failure_reason=None)
    policy = FRONT_TELE_POLICY
    scale_pass = inside_image_hull_area_px >= policy.minimum_inside_image_hull_area_px or projected_height_px >= policy.minimum_projected_height_px
    if not scale_pass:
        return GeometricObservabilityDecision(candidate=False, failure_reason=FAILURE_BELOW_PRIMARY_AREA_AND_HEIGHT)
    if inside_image_hull_ratio < policy.minimum_inside_image_hull_ratio:
        return GeometricObservabilityDecision(candidate=False, failure_reason=FAILURE_BELOW_MINIMUM_INSIDE_IMAGE_HULL_RATIO)
    return GeometricObservabilityDecision(candidate=True, failure_reason=None)

@dataclass(frozen=True, slots=True)
class CameraObservability:
    camera_name: str
    projection_valid: bool
    geometric_observability_candidate: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class ActorObservability:
    observability_format_version: str
    track_id: str
    actor_class: str
    observability_status: str
    visible_in_cameras: tuple[str, ...]
    camera_observability: tuple[CameraObservability, ...]
    actor_to_actor_occlusion_evaluated: bool
    static_occlusion_evaluated: bool

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value['visible_in_cameras'] = list(self.visible_in_cameras)
        value['camera_observability'] = [item.to_dict() for item in self.camera_observability]
        return value

def aggregate_actor_observability(projections_by_camera: Mapping[str, ActorCameraProjection]) -> ActorObservability:
    """Aggregate exactly four same-Actor projections in canonical camera order."""
    expected = set(CAMERA_NAMES)
    actual = set(projections_by_camera)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f'Expected exactly CAMERA_NAMES; missing={missing}, extra={extra}')
    projections = [projections_by_camera[name] for name in CAMERA_NAMES]
    track_ids = {projection.track_id for projection in projections}
    actor_classes = {projection.actor_class for projection in projections}
    if len(track_ids) != 1:
        raise ValueError('All projections must have the same track_id')
    if len(actor_classes) != 1:
        raise ValueError('All projections must have the same actor_class')
    camera_results: list[CameraObservability] = []
    visible_in_cameras: list[str] = []
    for expected_camera, projection in zip(CAMERA_NAMES, projections):
        if projection.camera_name != expected_camera:
            raise ValueError(f'Projection camera mismatch: key={expected_camera!r}, value={projection.camera_name!r}')
        if not projection.projection_valid:
            if not projection.failure_reason:
                raise ValueError('Invalid projection must provide a failure_reason')
            camera_results.append(CameraObservability(camera_name=expected_camera, projection_valid=False, geometric_observability_candidate=False, failure_reason=projection.failure_reason))
            continue
        if projection.failure_reason is not None:
            raise ValueError('Valid projection must not provide a failure_reason')
        decision = evaluate_geometric_observability(camera_name=expected_camera, inside_image_hull_area_px=projection.inside_image_hull_area_px, projected_height_px=projection.projected_height_px, inside_image_hull_ratio=projection.inside_image_hull_ratio)
        if decision.candidate:
            visible_in_cameras.append(expected_camera)
        camera_results.append(CameraObservability(camera_name=expected_camera, projection_valid=True, geometric_observability_candidate=decision.candidate, failure_reason=decision.failure_reason))
    status = 'candidate_visible' if visible_in_cameras else 'not_visible'
    return ActorObservability(observability_format_version=OBSERVABILITY_FORMAT_VERSION, track_id=next(iter(track_ids)), actor_class=next(iter(actor_classes)), observability_status=status, visible_in_cameras=tuple(visible_in_cameras), camera_observability=tuple(camera_results), actor_to_actor_occlusion_evaluated=False, static_occlusion_evaluated=False)

@dataclass(frozen=True, slots=True)
class MulticameraActorGeometricObservability:
    track_id: str
    actor_class: str
    projections: tuple[ActorCameraProjection, ...]
    observability: ActorObservability

@dataclass(frozen=True, slots=True)
class MulticameraActorGeometricObservabilityResult:
    actor_count: int
    actor_results: tuple[MulticameraActorGeometricObservability, ...]
    actor_observability: tuple[ActorObservability, ...]

def _track_id(actor: Mapping[str, Any]) -> str:
    value = actor.get('track_id')
    if value is None or str(value) == '':
        raise ValueError('Actor is missing a usable track_id')
    return str(value)

def _actor_class(actor: Mapping[str, Any]) -> str:
    value = actor.get('label_class')
    if value is None or str(value) == '':
        raise ValueError('Actor is missing a usable label_class')
    return str(value)

def build_multicamera_actor_geometric_observability(*, actors: Sequence[Mapping[str, Any]], recorded_ego_message: Mapping[str, Any], calibrations: Mapping[str, Any], samples_per_edge: int | None=None, near_plane_m: float=0.001, maximum_chord_error_px: float=1.0, maximum_adaptive_depth: int=14) -> MulticameraActorGeometricObservabilityResult:
    """Project and aggregate every Actor in deterministic track order."""
    expected_cameras = tuple(CAMERA_NAMES)
    expected_set = set(expected_cameras)
    actual_set = set(calibrations)
    if actual_set != expected_set:
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        raise ValueError(f'Expected exactly CAMERA_NAMES; missing={missing}, extra={extra}')
    source = tuple(actors)
    track_ids = tuple((_track_id(actor) for actor in source))
    if len(set(track_ids)) != len(track_ids):
        raise ValueError('Actor track_id values must be unique')
    for actor in source:
        _actor_class(actor)
    actor_results = []
    observability_results = []
    for actor in sorted(source, key=_track_id):
        track_id = _track_id(actor)
        actor_class = _actor_class(actor)
        projections_by_camera = {}
        for camera_name in expected_cameras:
            projection = project_actor_box_to_camera(actor, recorded_ego_message=recorded_ego_message, calibration=calibrations[camera_name], samples_per_edge=samples_per_edge, near_plane_m=near_plane_m, maximum_chord_error_px=maximum_chord_error_px, maximum_adaptive_depth=maximum_adaptive_depth)
            if projection.camera_name != camera_name:
                raise RuntimeError(f'Projection camera mismatch for track {track_id}: expected={camera_name}, actual={projection.camera_name}')
            if projection.track_id != track_id:
                raise RuntimeError(f'Projection track_id mismatch for track {track_id}')
            if projection.actor_class != actor_class:
                raise RuntimeError(f'Projection actor_class mismatch for track {track_id}')
            projections_by_camera[camera_name] = projection
        aggregated = aggregate_actor_observability(projections_by_camera)
        if aggregated.track_id != track_id:
            raise RuntimeError(f'Aggregated track_id mismatch for track {track_id}')
        if aggregated.actor_class != actor_class:
            raise RuntimeError(f'Aggregated actor_class mismatch for track {track_id}')
        ordered_projections = tuple((projections_by_camera[camera_name] for camera_name in expected_cameras))
        if tuple((item.camera_name for item in ordered_projections)) != expected_cameras:
            raise RuntimeError('Projection order is not canonical')
        actor_results.append(MulticameraActorGeometricObservability(track_id=track_id, actor_class=actor_class, projections=ordered_projections, observability=aggregated))
        observability_results.append(aggregated)
    return MulticameraActorGeometricObservabilityResult(actor_count=len(source), actor_results=tuple(actor_results), actor_observability=tuple(observability_results))
POLICY_NAME = 'step7e_dynamic_occlusion_v01'
MINIMUM_WINNING_CELL_COUNT = 2
MINIMUM_VISIBLE_FRACTION = 0.0

@dataclass(frozen=True, slots=True)
class ObservabilityShadowPolicy:
    policy_name: str
    minimum_winning_cell_count: int
    minimum_visible_fraction: float

    def __post_init__(self) -> None:
        if not self.policy_name:
            raise ValueError('policy_name must be non-empty')
        if isinstance(self.minimum_winning_cell_count, bool) or not isinstance(self.minimum_winning_cell_count, int) or self.minimum_winning_cell_count < 0:
            raise ValueError('minimum_winning_cell_count must be a non-negative integer')
        value = float(self.minimum_visible_fraction)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError('minimum_visible_fraction must be finite and within [0, 1]')

@dataclass(frozen=True, slots=True)
class ActorObservabilityShadowDecision:
    policy_name: str
    anchor_id: str
    track_id: str
    shadow_status: str
    winning_cell_count: int
    maximum_visible_fraction: float | None
    winning_cell_requirement_met: bool
    visible_fraction_requirement_met: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {'policy_name': self.policy_name, 'anchor_id': self.anchor_id, 'track_id': self.track_id, 'shadow_status': self.shadow_status, 'winning_cell_count': self.winning_cell_count, 'maximum_visible_fraction': self.maximum_visible_fraction, 'winning_cell_requirement_met': self.winning_cell_requirement_met, 'visible_fraction_requirement_met': self.visible_fraction_requirement_met, 'reasons': list(self.reasons)}

def evaluate_actor_observability_shadow(*, evidence: Mapping[str, Any], policy: ObservabilityShadowPolicy) -> ActorObservabilityShadowDecision:
    """Evaluate one policy against one exported Actor evidence row."""
    for field in ('anchor_id', 'track_id', 'evidence_status', 'total_winning_cell_count', 'maximum_visible_fraction', 'static_occlusion_evaluated'):
        if field not in evidence:
            raise ValueError(f'evidence is missing {field}')
    if bool(evidence['static_occlusion_evaluated']):
        raise ValueError('shadow evaluation expects static occlusion to remain unevaluated')
    winning_count = evidence['total_winning_cell_count']
    if isinstance(winning_count, bool) or not isinstance(winning_count, int) or winning_count < 0:
        raise ValueError('total_winning_cell_count must be a non-negative integer')
    visible_fraction = evidence['maximum_visible_fraction']
    if visible_fraction is not None:
        visible_fraction = float(visible_fraction)
        if not math.isfinite(visible_fraction) or not 0.0 <= visible_fraction <= 1.0:
            raise ValueError('maximum_visible_fraction must be finite and within [0, 1]')
    evidence_status = str(evidence['evidence_status'])
    winning_met = winning_count >= policy.minimum_winning_cell_count
    fraction_met = visible_fraction is not None and visible_fraction >= policy.minimum_visible_fraction
    if evidence_status == 'no_geometric_candidate':
        shadow_status = 'shadow_not_visible'
        reasons = ('no_geometric_candidate',)
        winning_met = False
        fraction_met = False
    elif evidence_status == 'candidate_without_sampled_surface':
        shadow_status = 'shadow_indeterminate'
        reasons = ('geometric_candidate_without_sampled_surface',)
        winning_met = False
        fraction_met = False
    elif evidence_status == 'combined_evidence_available':
        failures = []
        if not winning_met:
            failures.append('minimum_winning_cell_count_not_met')
        if not fraction_met:
            failures.append('minimum_visible_fraction_not_met')
        if failures:
            shadow_status = 'shadow_not_visible'
            reasons = tuple(failures)
        else:
            shadow_status = 'shadow_visible'
            reasons = ('dynamic_occlusion_policy_requirements_met',)
    else:
        raise ValueError(f'unexpected evidence_status: {evidence_status}')
    return ActorObservabilityShadowDecision(policy_name=policy.policy_name, anchor_id=str(evidence['anchor_id']), track_id=str(evidence['track_id']), shadow_status=shadow_status, winning_cell_count=winning_count, maximum_visible_fraction=visible_fraction, winning_cell_requirement_met=winning_met, visible_fraction_requirement_met=fraction_met, reasons=reasons)

def evaluate_actor_observability_shadow_policies(*, evidence_rows: Sequence[Mapping[str, Any]], policies: Sequence[ObservabilityShadowPolicy]) -> tuple[ActorObservabilityShadowDecision, ...]:
    """Evaluate all policies and rows in deterministic policy/Actor order."""
    policy_values = tuple(policies)
    names = tuple((policy.policy_name for policy in policy_values))
    if len(set(names)) != len(names):
        raise ValueError('shadow policy names must be unique')
    rows = tuple(evidence_rows)
    identities = tuple(((str(row.get('anchor_id', '')), str(row.get('track_id', ''))) for row in rows))
    if any((not anchor_id or not track_id for anchor_id, track_id in identities)):
        raise ValueError('evidence rows must have usable Anchor/Actor identities')
    if len(set(identities)) != len(identities):
        raise ValueError('evidence Anchor/Actor identities must be unique')
    ordered_rows = tuple((row for _, row in sorted(zip(identities, rows), key=lambda item: item[0])))
    return tuple((evaluate_actor_observability_shadow(evidence=row, policy=policy) for policy in policy_values for row in ordered_rows))
FROZEN_DYNAMIC_OCCLUSION_POLICY = ObservabilityShadowPolicy(policy_name=POLICY_NAME, minimum_winning_cell_count=MINIMUM_WINNING_CELL_COUNT, minimum_visible_fraction=MINIMUM_VISIBLE_FRACTION)

def evaluate_frozen_dynamic_occlusion_policy(*, evidence_rows: Sequence[Mapping[str, Any]]):
    """Evaluate the frozen policy without modifying source evidence."""
    return evaluate_actor_observability_shadow_policies(evidence_rows=tuple(evidence_rows), policies=(FROZEN_DYNAMIC_OCCLUSION_POLICY,))
