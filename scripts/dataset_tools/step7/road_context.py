"""Step 7G road, lane, wait-line, and intersection evidence domain.

Contains exact spatial indexing, Ego and Actor road matching, and summary contracts."""
from __future__ import annotations
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence
from step2.coordinate_utils import Point2D, Pose2D, normalize_angle, pose2d_from_pose_mapping
from step2.vector_map_reader import NearbyLane, VectorMapReader, point_in_polygon, project_point_to_polyline
from collections import Counter
LANE_SEARCH_RADIUS_M = 6.0
EGO_MAX_HEADING_ERROR_RAD = math.radians(70.0)
SPATIAL_CELL_SIZE_M = 20.0

@dataclass(frozen=True, slots=True)
class RoadLaneMatch:
    status: str
    lane_id: str | None
    centerline_distance_m: float | None
    centerline_arc_length_m: float | None
    lane_length_m: float | None
    inside_lane_polygon: bool | None
    heading_error_rad: float | None
    has_wait_line: bool | None
    wait_line_ids: tuple[str, ...]
    nearest_wait_line_id: str | None
    nearest_wait_line_type: str | None
    nearest_wait_line_distance_m: float | None
    nearest_wait_line_is_implicit: bool | None
    has_intersection_evidence: bool | None
    intersection_evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value['wait_line_ids'] = list(self.wait_line_ids)
        value['intersection_evidence'] = list(self.intersection_evidence)
        return value

class RoadFeatureMapContext:
    """One parsed map plus an exact candidate-preserving lane spatial index."""

    def __init__(self, *, raw_map: Mapping[str, Any], maximum_query_radius_m: float=LANE_SEARCH_RADIUS_M, cell_size_m: float=SPATIAL_CELL_SIZE_M) -> None:
        if raw_map.get('frame_id') != 'map':
            raise ValueError('VectorMap frame_id must be map')
        if maximum_query_radius_m <= 0.0 or cell_size_m <= 0.0:
            raise ValueError('spatial index dimensions must be positive')
        self.raw_map = raw_map
        self.vector_map = VectorMapReader.from_dict(raw_map)
        self.maximum_query_radius_m = float(maximum_query_radius_m)
        self.cell_size_m = float(cell_size_m)
        self.wait_lines_by_id = _wait_lines_by_id(raw_map)
        cells: dict[tuple[int, int], set[str]] = {}
        margin = self.maximum_query_radius_m
        for lane_id, lane in self.vector_map.lanes.items():
            xs = [point.x for point in lane.centerline]
            ys = [point.y for point in lane.centerline]
            minimum_x = min(xs) - margin
            maximum_x = max(xs) + margin
            minimum_y = min(ys) - margin
            maximum_y = max(ys) + margin
            for ix in range(self._cell(minimum_x), self._cell(maximum_x) + 1):
                for iy in range(self._cell(minimum_y), self._cell(maximum_y) + 1):
                    cells.setdefault((ix, iy), set()).add(lane_id)
        self._cells = {key: tuple(sorted(value)) for key, value in cells.items()}

    def _cell(self, value: float) -> int:
        return math.floor(value / self.cell_size_m)

    def candidate_lane_ids(self, point: Point2D, *, radius_m: float) -> tuple[str, ...]:
        if radius_m < 0.0:
            raise ValueError('radius_m must be non-negative')
        if radius_m > self.maximum_query_radius_m:
            raise ValueError('query radius exceeds spatial-index construction radius')
        return self._cells.get((self._cell(point.x), self._cell(point.y)), ())

    def find_nearby_lanes(self, point: Point2D, *, radius_m: float, yaw_rad: float | None=None, maximum_heading_error_rad: float | None=None, limit: int | None=None) -> tuple[NearbyLane, ...]:
        if maximum_heading_error_rad is not None and yaw_rad is None:
            raise ValueError('maximum_heading_error_rad requires yaw_rad')
        if limit is not None and limit <= 0:
            raise ValueError('limit must be positive')
        candidates = []
        for lane_id in self.candidate_lane_ids(point, radius_m=radius_m):
            lane = self.vector_map.require_lane(lane_id)
            projection = project_point_to_polyline(point, lane.centerline)
            if projection.distance_m > radius_m:
                continue
            heading_error = abs(normalize_angle(projection.heading_rad - yaw_rad)) if yaw_rad is not None else None
            if maximum_heading_error_rad is not None and heading_error is not None and (heading_error > maximum_heading_error_rad):
                continue
            candidates.append(NearbyLane(lane_id=lane_id, distance_m=projection.distance_m, heading_rad=projection.heading_rad, heading_error_rad=heading_error, inside_polygon=point_in_polygon(point, lane.polygon), projection=projection))
        candidates.sort(key=lambda candidate: (not candidate.inside_polygon, candidate.distance_m, candidate.heading_error_rad if candidate.heading_error_rad is not None else 0.0, candidate.lane_id))
        if limit is not None:
            candidates = candidates[:limit]
        return tuple(candidates)

def _wait_lines_by_id(raw_map: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for index, item in enumerate(raw_map.get('wait_lines', [])):
        if not isinstance(item, Mapping):
            raise TypeError(f'wait_lines[{index}] must be a mapping')
        wait_id = str(item.get('id', ''))
        if not wait_id or wait_id in result:
            raise ValueError('wait line ids must be non-empty and unique')
        result[wait_id] = item
    return result

def _wait_line_points(item: Mapping[str, Any]) -> tuple[Point2D, ...]:
    polyline = item.get('polyline')
    if not isinstance(polyline, Mapping):
        raise TypeError('wait line polyline must be a mapping')
    raw = polyline.get('points')
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError('wait line polyline requires at least two points')
    return tuple((Point2D(float(point['x']), float(point['y'])) for point in raw))

def match_point_to_road(*, context: RoadFeatureMapContext, point: Point2D, yaw_rad: float | None=None, maximum_heading_error_rad: float | None=None, radius_m: float=LANE_SEARCH_RADIUS_M) -> RoadLaneMatch:
    candidates = context.find_nearby_lanes(point, radius_m=radius_m, yaw_rad=yaw_rad, maximum_heading_error_rad=maximum_heading_error_rad, limit=1)
    if not candidates:
        return RoadLaneMatch('unmatched', None, None, None, None, None, None, None, (), None, None, None, None, None, ())
    candidate = candidates[0]
    lane = context.vector_map.require_lane(candidate.lane_id)
    available = tuple((wait_id for wait_id in lane.wait_line_ids if wait_id in context.wait_lines_by_id))
    nearest = None
    for wait_id in available:
        item = context.wait_lines_by_id[wait_id]
        projection = project_point_to_polyline(point, _wait_line_points(item))
        key = (projection.distance_m, wait_id)
        if nearest is None or key < nearest[0]:
            nearest = (key, item)
    evidence = []
    if available:
        evidence.append('wait_line')
    if len(context.vector_map.valid_related_lane_ids(lane.lane_id, 'successor')) > 1:
        evidence.append('multiple_successors')
    if len(context.vector_map.valid_related_lane_ids(lane.lane_id, 'predecessor')) > 1:
        evidence.append('multiple_predecessors')
    nearest_item = None if nearest is None else nearest[1]
    return RoadLaneMatch('matched', lane.lane_id, candidate.distance_m, candidate.projection.arc_length_m, lane.length_m, candidate.inside_polygon, candidate.heading_error_rad, bool(available), available, None if nearest_item is None else str(nearest_item['id']), None if nearest_item is None else str(nearest_item.get('wait_line_type', 'UNKNOWN')), None if nearest is None else nearest[0][0], None if nearest_item is None else bool(nearest_item.get('is_implicit', False)), bool(evidence), tuple(evidence))

def compute_ego_and_actor_road_features(*, context: RoadFeatureMapContext, ego_pose: Pose2D, actors: Sequence[Mapping[str, Any]]) -> tuple[RoadLaneMatch, tuple[tuple[str, str, RoadLaneMatch, str], ...]]:
    ego = match_point_to_road(context=context, point=Point2D(ego_pose.x, ego_pose.y), yaw_rad=ego_pose.yaw, maximum_heading_error_rad=EGO_MAX_HEADING_ERROR_RAD)
    rows = []
    seen = set()
    for actor in actors:
        track_id = str(actor.get('track_id', ''))
        label = str(actor.get('label_class', ''))
        if not track_id or track_id in seen:
            raise ValueError('Actor track ids must be unique and non-empty')
        seen.add(track_id)
        pose = pose2d_from_pose_mapping(actor['pose'])
        match = match_point_to_road(context=context, point=Point2D(pose.x, pose.y))
        relation = 'unknown' if ego.lane_id is None or match.lane_id is None else context.vector_map.relation(ego.lane_id, match.lane_id)
        rows.append((track_id, label, match, relation))
    return (ego, tuple(sorted(rows, key=lambda item: item[0])))

def summarize_road_feature_rows(*, keyframes, actor_rows, ego_rows):
    anchors = [str(x['anchor_id']) for x in keyframes]
    if len(anchors) != len(set(anchors)):
        raise ValueError('keyframe anchors must be unique')
    if len(ego_rows) != len(anchors):
        raise ValueError('exactly one Ego road row per Keyframe is required')
    actor_ids = [(str(x['anchor_id']), str(x['track_id'])) for x in actor_rows]
    if len(actor_ids) != len(set(actor_ids)):
        raise ValueError('Actor road identities must be unique')
    return {'schema_version': 'step7g-road-features-summary-v01', 'keyframe_count': len(anchors), 'ego_row_count': len(ego_rows), 'actor_row_count': len(actor_rows), 'ego_match_status_counts': dict(sorted(Counter((x['lane_match_status'] for x in ego_rows)).items())), 'actor_match_status_counts': dict(sorted(Counter((x['lane_match_status'] for x in actor_rows)).items())), 'actor_ego_lane_relation_counts': dict(sorted(Counter((x['ego_lane_relation'] for x in actor_rows)).items())), 'ego_intersection_evidence_count': sum((bool(x['has_intersection_evidence']) for x in ego_rows)), 'actor_intersection_evidence_count': sum((bool(x['has_intersection_evidence']) for x in actor_rows))}
