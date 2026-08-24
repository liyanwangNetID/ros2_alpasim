#!/usr/bin/env python3
"""Read and query AlpaSim VectorMap lane geometry and topology.

This module is independent of ROS 2. It parses the JSON structure stored in
map/vector_map.json and exposes deterministic geometry/topology operations for
lane matching and map-driven meta-action generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from coordinate_utils import Point2D, normalize_angle


_EPSILON = 1e-12


class VectorMapError(ValueError):
    """Base exception for malformed VectorMap data."""


@dataclass(frozen=True, slots=True)
class PolylineProjection:
    point: Point2D
    distance_m: float
    heading_rad: float
    segment_index: int
    segment_ratio: float
    arc_length_m: float


@dataclass(frozen=True, slots=True)
class Lane:
    lane_id: str
    centerline: tuple[Point2D, ...]
    centerline_headings: tuple[float, ...]
    left_boundary: tuple[Point2D, ...]
    right_boundary: tuple[Point2D, ...]
    polygon: tuple[Point2D, ...]
    predecessor_ids: tuple[str, ...]
    successor_ids: tuple[str, ...]
    left_adjacent_ids: tuple[str, ...]
    right_adjacent_ids: tuple[str, ...]
    road_area_ids: tuple[str, ...]
    traffic_sign_ids: tuple[str, ...]
    wait_line_ids: tuple[str, ...]

    @property
    def start_point(self) -> Point2D:
        return self.centerline[0]

    @property
    def end_point(self) -> Point2D:
        return self.centerline[-1]

    @property
    def length_m(self) -> float:
        return polyline_length(self.centerline)


@dataclass(frozen=True, slots=True)
class NearbyLane:
    lane_id: str
    distance_m: float
    heading_rad: float
    heading_error_rad: float | None
    inside_polygon: bool
    projection: PolylineProjection


@dataclass(frozen=True, slots=True)
class TopologyWarning:
    source_lane_id: str
    relation: str
    missing_lane_id: str


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VectorMapError(f"{field_name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise VectorMapError(f"{field_name} must be finite")
    return number


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return tuple()
    if not isinstance(value, list):
        raise VectorMapError(f"{field_name} must be a list")
    return tuple(str(item) for item in value)


def _parse_points(value: Any, field_name: str) -> tuple[Point2D, ...]:
    if not isinstance(value, list):
        raise VectorMapError(f"{field_name} must be a list")
    points: list[Point2D] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise VectorMapError(f"{field_name}[{index}] must be an object")
        points.append(
            Point2D(
                x=_finite_float(item.get("x"), f"{field_name}[{index}].x"),
                y=_finite_float(item.get("y"), f"{field_name}[{index}].y"),
            )
        )
    return tuple(points)


def _parse_polyline(
    value: Any,
    field_name: str,
    *,
    minimum_points: int,
) -> tuple[tuple[Point2D, ...], tuple[float, ...]]:
    if not isinstance(value, Mapping):
        raise VectorMapError(f"{field_name} must be an object")
    points = _parse_points(value.get("points"), f"{field_name}.points")
    if len(points) < minimum_points:
        raise VectorMapError(
            f"{field_name} requires at least {minimum_points} points"
        )
    raw_headings = value.get("headings", [])
    if not isinstance(raw_headings, list):
        raise VectorMapError(f"{field_name}.headings must be a list")
    headings = tuple(
        normalize_angle(_finite_float(item, f"{field_name}.headings"))
        for item in raw_headings
    )
    return points, headings


def build_lane_polygon(
    left_boundary: Sequence[Point2D],
    right_boundary: Sequence[Point2D],
) -> tuple[Point2D, ...]:
    """Build a closed lane polygon from ordered left and right boundaries."""
    if len(left_boundary) < 2 or len(right_boundary) < 2:
        return tuple()
    polygon = list(left_boundary) + list(reversed(right_boundary))
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return tuple(polygon)


def polyline_length(points: Sequence[Point2D]) -> float:
    return sum(
        math.hypot(current.x - previous.x, current.y - previous.y)
        for previous, current in zip(points, points[1:])
    )


def point_on_segment(
    point: Point2D,
    start: Point2D,
    end: Point2D,
    *,
    tolerance_m: float = 1e-8,
) -> bool:
    dx = end.x - start.x
    dy = end.y - start.y
    length_squared = dx * dx + dy * dy
    if length_squared <= _EPSILON:
        return math.hypot(point.x - start.x, point.y - start.y) <= tolerance_m
    ratio = (
        (point.x - start.x) * dx + (point.y - start.y) * dy
    ) / length_squared
    if ratio < 0.0 or ratio > 1.0:
        return False
    projected_x = start.x + ratio * dx
    projected_y = start.y + ratio * dy
    return math.hypot(point.x - projected_x, point.y - projected_y) <= tolerance_m


def point_in_polygon(point: Point2D, polygon: Sequence[Point2D]) -> bool:
    """Return True for points inside or on the boundary of a simple polygon."""
    if len(polygon) < 4:
        return False
    inside = False
    vertices = polygon[:-1] if polygon[0] == polygon[-1] else polygon
    previous = vertices[-1]
    for current in vertices:
        if point_on_segment(point, previous, current):
            return True
        crosses = (current.y > point.y) != (previous.y > point.y)
        if crosses:
            intersection_x = (
                previous.x
                + (point.y - previous.y)
                * (current.x - previous.x)
                / (current.y - previous.y)
            )
            if point.x < intersection_x:
                inside = not inside
        previous = current
    return inside


def project_point_to_polyline(
    point: Point2D,
    polyline: Sequence[Point2D],
) -> PolylineProjection:
    if len(polyline) < 2:
        raise VectorMapError("polyline requires at least two points")

    best: PolylineProjection | None = None
    cumulative = 0.0
    for segment_index, (start, end) in enumerate(
        zip(polyline, polyline[1:])
    ):
        dx = end.x - start.x
        dy = end.y - start.y
        length_squared = dx * dx + dy * dy
        if length_squared <= _EPSILON:
            continue
        segment_length = math.sqrt(length_squared)
        ratio = (
            (point.x - start.x) * dx + (point.y - start.y) * dy
        ) / length_squared
        ratio = min(1.0, max(0.0, ratio))
        projected = Point2D(
            x=start.x + ratio * dx,
            y=start.y + ratio * dy,
        )
        distance = math.hypot(
            point.x - projected.x,
            point.y - projected.y,
        )
        candidate = PolylineProjection(
            point=projected,
            distance_m=distance,
            heading_rad=math.atan2(dy, dx),
            segment_index=segment_index,
            segment_ratio=ratio,
            arc_length_m=cumulative + ratio * segment_length,
        )
        if best is None or (
            candidate.distance_m,
            candidate.segment_index,
            candidate.segment_ratio,
        ) < (
            best.distance_m,
            best.segment_index,
            best.segment_ratio,
        ):
            best = candidate
        cumulative += segment_length

    if best is None:
        raise VectorMapError("polyline contains no positive-length segment")
    return best


class VectorMapReader:
    """Parsed lane geometry and topology for one AlpaSim VectorMap."""

    def __init__(
        self,
        *,
        frame_id: str,
        map_id: str,
        revision: int | None,
        lanes: Mapping[str, Lane],
        topology_warnings: Sequence[TopologyWarning],
        road_edges: Sequence[dict[str, Any]],
        traffic_signs: Sequence[dict[str, Any]],
        wait_lines: Sequence[dict[str, Any]],
    ) -> None:
        self.frame_id = frame_id
        self.map_id = map_id
        self.revision = revision
        self._lanes = dict(lanes)
        self.topology_warnings = tuple(topology_warnings)
        self.road_edges = tuple(road_edges)
        self.traffic_signs = tuple(traffic_signs)
        self.wait_lines = tuple(wait_lines)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VectorMapReader":
        raw_lanes = data.get("lanes")
        if not isinstance(raw_lanes, list):
            raise VectorMapError("VectorMap lanes must be a list")

        lanes: dict[str, Lane] = {}
        for index, raw_lane in enumerate(raw_lanes):
            if not isinstance(raw_lane, Mapping):
                raise VectorMapError(f"lanes[{index}] must be an object")
            lane_id = str(raw_lane.get("id", ""))
            if not lane_id:
                raise VectorMapError(f"lanes[{index}] has an empty id")
            if lane_id in lanes:
                raise VectorMapError(f"duplicate lane id: {lane_id}")

            centerline, centerline_headings = _parse_polyline(
                raw_lane.get("centerline"),
                f"lanes[{index}].centerline",
                minimum_points=2,
            )
            left_boundary, _ = _parse_polyline(
                raw_lane.get("left_boundary"),
                f"lanes[{index}].left_boundary",
                minimum_points=2,
            )
            right_boundary, _ = _parse_polyline(
                raw_lane.get("right_boundary"),
                f"lanes[{index}].right_boundary",
                minimum_points=2,
            )
            lane = Lane(
                lane_id=lane_id,
                centerline=centerline,
                centerline_headings=centerline_headings,
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                polygon=build_lane_polygon(left_boundary, right_boundary),
                predecessor_ids=_string_tuple(
                    raw_lane.get("predecessor_ids", []),
                    f"lanes[{index}].predecessor_ids",
                ),
                successor_ids=_string_tuple(
                    raw_lane.get("successor_ids", []),
                    f"lanes[{index}].successor_ids",
                ),
                left_adjacent_ids=_string_tuple(
                    raw_lane.get("left_adjacent_ids", []),
                    f"lanes[{index}].left_adjacent_ids",
                ),
                right_adjacent_ids=_string_tuple(
                    raw_lane.get("right_adjacent_ids", []),
                    f"lanes[{index}].right_adjacent_ids",
                ),
                road_area_ids=_string_tuple(
                    raw_lane.get("road_area_ids", []),
                    f"lanes[{index}].road_area_ids",
                ),
                traffic_sign_ids=_string_tuple(
                    raw_lane.get("traffic_sign_ids", []),
                    f"lanes[{index}].traffic_sign_ids",
                ),
                wait_line_ids=_string_tuple(
                    raw_lane.get("wait_line_ids", []),
                    f"lanes[{index}].wait_line_ids",
                ),
            )
            lanes[lane_id] = lane

        warnings: list[TopologyWarning] = []
        relation_fields = (
            ("predecessor", "predecessor_ids"),
            ("successor", "successor_ids"),
            ("left_adjacent", "left_adjacent_ids"),
            ("right_adjacent", "right_adjacent_ids"),
        )
        for lane in lanes.values():
            for relation_name, attribute in relation_fields:
                for related_id in getattr(lane, attribute):
                    if related_id not in lanes:
                        warnings.append(
                            TopologyWarning(
                                source_lane_id=lane.lane_id,
                                relation=relation_name,
                                missing_lane_id=related_id,
                            )
                        )

        return cls(
            frame_id=str(data.get("frame_id", "")),
            map_id=str(data.get("map_id", "")),
            revision=(
                int(data["revision"])
                if data.get("revision") is not None
                else None
            ),
            lanes=lanes,
            topology_warnings=warnings,
            road_edges=tuple(
                item for item in data.get("road_edges", [])
                if isinstance(item, dict)
            ),
            traffic_signs=tuple(
                item for item in data.get("traffic_signs", [])
                if isinstance(item, dict)
            ),
            wait_lines=tuple(
                item for item in data.get("wait_lines", [])
                if isinstance(item, dict)
            ),
        )

    @property
    def lanes(self) -> Mapping[str, Lane]:
        return self._lanes

    def __len__(self) -> int:
        return len(self._lanes)

    def get_lane(self, lane_id: str) -> Lane | None:
        return self._lanes.get(str(lane_id))

    def require_lane(self, lane_id: str) -> Lane:
        lane = self.get_lane(lane_id)
        if lane is None:
            raise KeyError(f"unknown lane id: {lane_id}")
        return lane

    def relation(self, source_lane_id: str, target_lane_id: str) -> str:
        """Return the direct topology relation from source to target."""
        source = self.require_lane(source_lane_id)
        target = str(target_lane_id)
        if target == source.lane_id:
            return "same"
        if target in source.successor_ids:
            return "successor"
        if target in source.predecessor_ids:
            return "predecessor"
        if target in source.left_adjacent_ids:
            return "left_adjacent"
        if target in source.right_adjacent_ids:
            return "right_adjacent"
        return "unrelated"

    def valid_related_lane_ids(
        self,
        lane_id: str,
        relation: str,
    ) -> tuple[str, ...]:
        lane = self.require_lane(lane_id)
        attribute_by_relation = {
            "predecessor": "predecessor_ids",
            "successor": "successor_ids",
            "left_adjacent": "left_adjacent_ids",
            "right_adjacent": "right_adjacent_ids",
        }
        if relation not in attribute_by_relation:
            raise ValueError(f"unsupported relation: {relation}")
        return tuple(
            related_id
            for related_id in getattr(
                lane,
                attribute_by_relation[relation],
            )
            if related_id in self._lanes
        )

    def follow_successors(
        self,
        start_lane_id: str,
        *,
        maximum_depth: int = 10,
    ) -> tuple[str, ...]:
        """Follow a unique-successor corridor until a branch, end, or cycle."""
        if maximum_depth < 0:
            raise ValueError("maximum_depth must be non-negative")
        lane_id = self.require_lane(start_lane_id).lane_id
        result = [lane_id]
        visited = {lane_id}
        for _ in range(maximum_depth):
            successors = self.valid_related_lane_ids(lane_id, "successor")
            if len(successors) != 1:
                break
            next_lane_id = successors[0]
            if next_lane_id in visited:
                break
            result.append(next_lane_id)
            visited.add(next_lane_id)
            lane_id = next_lane_id
        return tuple(result)

    def project_to_lane(
        self,
        lane_id: str,
        point: Point2D,
    ) -> PolylineProjection:
        lane = self.require_lane(lane_id)
        return project_point_to_polyline(point, lane.centerline)

    def lane_contains_point(self, lane_id: str, point: Point2D) -> bool:
        lane = self.require_lane(lane_id)
        return point_in_polygon(point, lane.polygon)

    def find_nearby_lanes(
        self,
        x: float,
        y: float,
        *,
        radius_m: float,
        yaw_rad: float | None = None,
        maximum_heading_error_rad: float | None = None,
        limit: int | None = None,
    ) -> tuple[NearbyLane, ...]:
        if radius_m < 0.0:
            raise ValueError("radius_m must be non-negative")
        if maximum_heading_error_rad is not None and yaw_rad is None:
            raise ValueError(
                "maximum_heading_error_rad requires yaw_rad"
            )
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")

        point = Point2D(float(x), float(y))
        candidates: list[NearbyLane] = []
        for lane in self._lanes.values():
            projection = project_point_to_polyline(point, lane.centerline)
            if projection.distance_m > radius_m:
                continue
            heading_error = (
                abs(normalize_angle(projection.heading_rad - yaw_rad))
                if yaw_rad is not None
                else None
            )
            if (
                maximum_heading_error_rad is not None
                and heading_error is not None
                and heading_error > maximum_heading_error_rad
            ):
                continue
            candidates.append(
                NearbyLane(
                    lane_id=lane.lane_id,
                    distance_m=projection.distance_m,
                    heading_rad=projection.heading_rad,
                    heading_error_rad=heading_error,
                    inside_polygon=point_in_polygon(point, lane.polygon),
                    projection=projection,
                )
            )

        candidates.sort(
            key=lambda candidate: (
                not candidate.inside_polygon,
                candidate.distance_m,
                candidate.heading_error_rad
                if candidate.heading_error_rad is not None
                else 0.0,
                candidate.lane_id,
            )
        )
        if limit is not None:
            candidates = candidates[:limit]
        return tuple(candidates)
