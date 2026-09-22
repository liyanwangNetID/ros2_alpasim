"""Step 7 triangle, box-surface, FOV, clipping, and perspective-depth geometry domain."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Protocol, Sequence
from step7.projection import PixelProjection, Vector3
from typing import Sequence
from step7.projection import Vector3
from step7.projection import clip_segment_to_angular_fov
from typing import Literal, Sequence
from typing import Iterable, Sequence
from typing import Final, Sequence

class CameraPointProjector(Protocol):

    def project_camera_point(self, point: Vector3) -> PixelProjection:
        ...

@dataclass(frozen=True, slots=True)
class AngularFovBoundaryProjectedExtent:
    camera_points: tuple[Vector3, ...]
    projections: tuple[PixelProjection, ...]
    point_count: int
    width_px: float | None
    height_px: float | None
    maximum_pair_distance_px: float | None
    has_measurable_extent: bool

def _extent_points_close(first: Vector3, second: Vector3) -> bool:
    return all((math.isclose(first_value, second_value, rel_tol=1e-12, abs_tol=1e-12) for first_value, second_value in ((first.x, second.x), (first.y, second.y), (first.z, second.z))))

def _extent_append_unique_point(points: list[Vector3], candidate: Vector3) -> None:
    if not any((_extent_points_close(existing, candidate) for existing in points)):
        points.append(candidate)

def _extent_validate_point(point: Vector3) -> None:
    if not all((math.isfinite(value) for value in (point.x, point.y, point.z))):
        raise ValueError('triangle coordinates must be finite')
    if point.z <= 0.0:
        raise ValueError('triangle vertices must have positive depth')

def summarize_angular_fov_boundary_projected_extent(triangle_camera: Sequence[Vector3], calibration: CameraPointProjector, *, max_angle_rad: float | None) -> AngularFovBoundaryProjectedExtent:
    """Project unique in-FOV vertex and edge-intersection evidence points."""
    if len(triangle_camera) != 3:
        raise ValueError('a triangle must contain exactly three vertices')
    vertices = tuple(triangle_camera)
    for vertex in vertices:
        _extent_validate_point(vertex)
    diagnostics = classify_triangle_angular_fov(vertices, max_angle_rad=max_angle_rad)
    unique_points: list[Vector3] = []
    for vertex, inside in zip(vertices, diagnostics.vertex_inside):
        if inside:
            _extent_append_unique_point(unique_points, vertex)
    for edge in diagnostics.edge_intersections:
        if edge.clipped_segment is None:
            continue
        for point in edge.clipped_segment:
            _extent_append_unique_point(unique_points, point)
    points = tuple(unique_points)
    projections_list: list[PixelProjection] = []
    for point in points:
        projection = calibration.project_camera_point(point)
        if not projection.positive_z:
            raise ValueError('evidence point projection must have positive depth')
        if not projection.within_fov:
            raise ValueError('evidence point projection must be within FOV')
        if not math.isfinite(projection.u) or not math.isfinite(projection.v):
            raise ValueError('evidence point pixel coordinates must be finite')
        projections_list.append(projection)
    projections = tuple(projections_list)
    if not projections:
        return AngularFovBoundaryProjectedExtent(camera_points=points, projections=projections, point_count=0, width_px=None, height_px=None, maximum_pair_distance_px=None, has_measurable_extent=False)
    u_values = [item.u for item in projections]
    v_values = [item.v for item in projections]
    maximum_distance = 0.0
    for first_index, first in enumerate(projections):
        for second in projections[first_index + 1:]:
            maximum_distance = max(maximum_distance, math.hypot(second.u - first.u, second.v - first.v))
    return AngularFovBoundaryProjectedExtent(camera_points=points, projections=projections, point_count=len(points), width_px=max(u_values) - min(u_values), height_px=max(v_values) - min(v_values), maximum_pair_distance_px=maximum_distance, has_measurable_extent=True)
Triangle3D = tuple[Vector3, Vector3, Vector3]

@dataclass(frozen=True, slots=True)
class SafeAngularFovPolygon:
    ordered_vertices_camera: tuple[Vector3, ...]
    triangles_camera: tuple[Triangle3D, ...]
    has_measurable_polygon: bool
    is_degenerate: bool

def _safe_subtract(first: Vector3, second: Vector3) -> tuple[float, float, float]:
    return (first.x - second.x, first.y - second.y, first.z - second.z)

def _safe_dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return sum((a * b for a, b in zip(first, second)))

def _safe_norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_safe_dot(vector, vector))

def _safe_normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    length = _safe_norm(vector)
    if length <= 1e-15:
        raise ValueError('source triangle must be non-degenerate')
    return tuple((value / length for value in vector))

def _safe_points_close(first: Vector3, second: Vector3, tolerance: float) -> bool:
    return all((math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance) for a, b in ((first.x, second.x), (first.y, second.y), (first.z, second.z))))

def build_safe_angular_fov_polygon(triangle_camera: Sequence[Vector3], *, max_angle_rad: float | None, point_tolerance: float=1e-12) -> SafeAngularFovPolygon:
    """Return ordered in-FOV evidence polygon and fan triangulation."""
    if len(triangle_camera) != 3:
        raise ValueError('a triangle must contain exactly three vertices')
    tolerance = float(point_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('point_tolerance must be positive and finite')
    vertices: Triangle3D = tuple(triangle_camera)
    for point in vertices:
        if not all((math.isfinite(value) for value in (point.x, point.y, point.z))):
            raise ValueError('triangle coordinates must be finite')
        if point.z <= 0.0:
            raise ValueError('triangle vertices must have positive depth')
    first, second, third = vertices
    axis_u = _safe_normalize(_safe_subtract(second, first))
    third_offset = _safe_subtract(third, first)
    orthogonal = tuple((value - _safe_dot(third_offset, axis_u) * axis for value, axis in zip(third_offset, axis_u)))
    axis_v = _safe_normalize(orthogonal)
    diagnostics = classify_triangle_angular_fov(vertices, max_angle_rad=max_angle_rad)
    unique: list[Vector3] = []

    def append_unique(candidate: Vector3) -> None:
        if not any((_safe_points_close(existing, candidate, tolerance) for existing in unique)):
            unique.append(candidate)
    for point, inside in zip(vertices, diagnostics.vertex_inside):
        if inside:
            append_unique(point)
    for edge in diagnostics.edge_intersections:
        if edge.clipped_segment is not None:
            for point in edge.clipped_segment:
                append_unique(point)
    if not unique:
        return SafeAngularFovPolygon((), (), False, False)
    if len(unique) < 3:
        return SafeAngularFovPolygon(tuple(unique), (), True, True)
    centroid = Vector3(sum((point.x for point in unique)) / len(unique), sum((point.y for point in unique)) / len(unique), sum((point.z for point in unique)) / len(unique))

    def angle(point: Vector3) -> float:
        offset = _safe_subtract(point, centroid)
        return math.atan2(_safe_dot(offset, axis_v), _safe_dot(offset, axis_u))
    ordered = tuple(sorted(unique, key=angle))
    anchor = ordered[0]
    triangles = tuple(((anchor, ordered[index], ordered[index + 1]) for index in range(1, len(ordered) - 1)))
    return SafeAngularFovPolygon(ordered, triangles, True, False)
FOV_EDGE_INDEX_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (1, 2), (2, 0))
_EPSILON = 1e-12

@dataclass(frozen=True, slots=True)
class TriangleEdgeFovIntersection:
    first_vertex_index: int
    second_vertex_index: int
    clipped_segment: tuple[Vector3, Vector3] | None

@dataclass(frozen=True, slots=True)
class TriangleAngularFovDiagnostics:
    vertex_inside: tuple[bool, bool, bool]
    edge_intersections: tuple[TriangleEdgeFovIntersection, ...]
    all_vertices_inside: bool
    any_vertex_inside: bool
    every_edge_has_no_fov_segment: bool

def _diag_validate_max_angle(max_angle_rad: float | None) -> float | None:
    if max_angle_rad is None:
        return None
    maximum = float(max_angle_rad)
    if not math.isfinite(maximum) or not 0.0 < maximum < math.pi / 2.0:
        raise ValueError('max_angle_rad must be in (0, pi/2)')
    return maximum

def point_is_within_angular_fov(point: Vector3, *, max_angle_rad: float | None) -> bool:
    """Return whether a positive-depth camera point lies in the angular cone.

    The cone boundary is included using the same numerical tolerance as the
    existing analytic segment clipper. When max_angle_rad is None, every
    positive-depth finite point is accepted.
    """
    values = (point.x, point.y, point.z)
    if not all((math.isfinite(value) for value in values)):
        raise ValueError('point coordinates must be finite')
    if point.z <= 0.0:
        return False
    maximum = _diag_validate_max_angle(max_angle_rad)
    if maximum is None:
        return True
    tangent_squared = math.tan(maximum) ** 2
    cone_value = point.x * point.x + point.y * point.y - tangent_squared * point.z * point.z
    scale = max(1.0, point.x * point.x + point.y * point.y, tangent_squared * point.z * point.z)
    return cone_value <= _EPSILON * scale

def classify_triangle_angular_fov(triangle_camera: Sequence[Vector3], *, max_angle_rad: float | None) -> TriangleAngularFovDiagnostics:
    """Return conservative vertex and edge diagnostics for one triangle.

    `every_edge_has_no_fov_segment` must not be interpreted as proof that the
    triangle interior misses the cone. Exact surface classification requires
    interior sampling or adaptive triangle subdivision.
    """
    if len(triangle_camera) != 3:
        raise ValueError('a triangle must contain exactly three vertices')
    maximum = _diag_validate_max_angle(max_angle_rad)
    vertices = tuple(triangle_camera)
    vertex_inside = tuple((point_is_within_angular_fov(point, max_angle_rad=maximum) for point in vertices))
    edges = tuple((TriangleEdgeFovIntersection(first_vertex_index=first_index, second_vertex_index=second_index, clipped_segment=clip_segment_to_angular_fov(vertices[first_index], vertices[second_index], max_angle_rad=maximum)) for first_index, second_index in FOV_EDGE_INDEX_PAIRS))
    return TriangleAngularFovDiagnostics(vertex_inside=vertex_inside, edge_intersections=edges, all_vertices_inside=all(vertex_inside), any_vertex_inside=any(vertex_inside), every_edge_has_no_fov_segment=all((edge.clipped_segment is None for edge in edges)))

@dataclass(frozen=True, slots=True)
class AngularFovAcceptedTriangle:
    vertices_camera: Triangle3D
    subdivision_depth: int

@dataclass(frozen=True, slots=True)
class AngularFovBoundaryTriangle:
    vertices_camera: Triangle3D
    subdivision_depth: int
    sample_inside_count: int
    edge_intersection_count: int
    maximum_boundary_extent_px: float | None

@dataclass(frozen=True, slots=True)
class AngularFovTriangleSubdivision:
    accepted_inside_triangles: tuple[AngularFovAcceptedTriangle, ...]
    boundary_approximated_triangles: tuple[AngularFovBoundaryTriangle, ...]
    boundary_depth_limited_triangles: tuple[AngularFovBoundaryTriangle, ...]
    boundary_unmeasurable_triangles: tuple[AngularFovBoundaryTriangle, ...]
    rejected_outside_triangle_count: int
    maximum_depth_reached: int
    stopped_by_depth_limit: bool

    @property
    def boundary_unresolved_triangles(self) -> tuple[AngularFovBoundaryTriangle, ...]:
        return self.boundary_depth_limited_triangles + self.boundary_unmeasurable_triangles

def _subdiv_finite_point(point: Vector3) -> None:
    if not all((math.isfinite(value) for value in (point.x, point.y, point.z))):
        raise ValueError('triangle coordinates must be finite')
    if point.z <= 0.0:
        raise ValueError('triangle vertices must have positive depth')

def _subdiv_midpoint(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(0.5 * (first.x + second.x), 0.5 * (first.y + second.y), 0.5 * (first.z + second.z))

def _subdiv_centroid(first: Vector3, second: Vector3, third: Vector3) -> Vector3:
    return Vector3((first.x + second.x + third.x) / 3.0, (first.y + second.y + third.y) / 3.0, (first.z + second.z + third.z) / 3.0)

def subdivide_triangle_to_angular_fov(triangle_camera: Sequence[Vector3], *, max_angle_rad: float | None, maximum_depth: int, calibration=None, maximum_boundary_extent_px: float | None=None) -> AngularFovTriangleSubdivision:
    """Recursively classify a positive-depth triangle against the FOV cone.

    A leaf is accepted only when its three vertices, three edge midpoints, and
    centroid are all inside. At the depth limit, every non-accepted leaf is
    reported as unresolved. No unresolved leaf is silently treated as inside.

    Fully outside leaves are counted as rejected only when all seven samples
    are outside and analytic checks show no FOV segment on any edge. Before the
    depth limit such leaves are conservatively subdivided because their
    interiors may still intersect the cone.
    """
    if len(triangle_camera) != 3:
        raise ValueError('a triangle must contain exactly three vertices')
    if isinstance(maximum_depth, bool) or not isinstance(maximum_depth, int):
        raise TypeError('maximum_depth must be an integer')
    if maximum_depth < 0:
        raise ValueError('maximum_depth must be non-negative')
    if maximum_boundary_extent_px is not None:
        boundary_limit = float(maximum_boundary_extent_px)
        if not math.isfinite(boundary_limit) or boundary_limit <= 0.0:
            raise ValueError('maximum_boundary_extent_px must be positive and finite')
        if calibration is None:
            raise ValueError('calibration is required when maximum_boundary_extent_px is set')
    else:
        boundary_limit = None
    root: Triangle3D = tuple(triangle_camera)
    for point in root:
        _subdiv_finite_point(point)
    accepted: list[AngularFovAcceptedTriangle] = []
    approximated: list[AngularFovBoundaryTriangle] = []
    depth_limited: list[AngularFovBoundaryTriangle] = []
    unmeasurable: list[AngularFovBoundaryTriangle] = []
    rejected_count = 0
    maximum_depth_reached = 0

    def recurse(vertices: Triangle3D, depth: int) -> None:
        nonlocal rejected_count
        nonlocal maximum_depth_reached
        maximum_depth_reached = max(maximum_depth_reached, depth)
        if max_angle_rad is not None:
            intersection = triangle_intersects_angular_fov_cone(vertices, max_angle_rad=max_angle_rad)
            if not intersection.intersects:
                rejected_count += 1
                return
        first, second, third = vertices
        midpoint_ab = _subdiv_midpoint(first, second)
        midpoint_bc = _subdiv_midpoint(second, third)
        midpoint_ca = _subdiv_midpoint(third, first)
        centroid = _subdiv_centroid(first, second, third)
        samples = (first, second, third, midpoint_ab, midpoint_bc, midpoint_ca, centroid)
        sample_inside = tuple((point_is_within_angular_fov(point, max_angle_rad=max_angle_rad) for point in samples))
        inside_count = sum(sample_inside)
        if inside_count == len(samples):
            accepted.append(AngularFovAcceptedTriangle(vertices_camera=vertices, subdivision_depth=depth))
            return
        diagnostics = classify_triangle_angular_fov(vertices, max_angle_rad=max_angle_rad)
        edge_intersection_count = sum((edge.clipped_segment is not None for edge in diagnostics.edge_intersections))
        boundary_extent = None
        measurable = False
        if boundary_limit is not None:
            extent = summarize_angular_fov_boundary_projected_extent(vertices, calibration, max_angle_rad=max_angle_rad)
            measurable = extent.has_measurable_extent
            boundary_extent = extent.maximum_pair_distance_px
            if measurable and boundary_extent is not None and (boundary_extent <= boundary_limit):
                approximated.append(AngularFovBoundaryTriangle(vertices_camera=vertices, subdivision_depth=depth, sample_inside_count=inside_count, edge_intersection_count=edge_intersection_count, maximum_boundary_extent_px=boundary_extent))
                return
        if depth >= maximum_depth:
            boundary = AngularFovBoundaryTriangle(vertices_camera=vertices, subdivision_depth=depth, sample_inside_count=inside_count, edge_intersection_count=edge_intersection_count, maximum_boundary_extent_px=boundary_extent)
            if boundary_limit is not None and (not measurable):
                unmeasurable.append(boundary)
            else:
                depth_limited.append(boundary)
            return
        children: tuple[Triangle3D, ...] = ((first, midpoint_ab, midpoint_ca), (midpoint_ab, second, midpoint_bc), (midpoint_ca, midpoint_bc, third), (midpoint_ab, midpoint_bc, midpoint_ca))
        for child in children:
            recurse(child, depth + 1)
    recurse(root, 0)
    return AngularFovTriangleSubdivision(accepted_inside_triangles=tuple(accepted), boundary_approximated_triangles=tuple(approximated), boundary_depth_limited_triangles=tuple(depth_limited), boundary_unmeasurable_triangles=tuple(unmeasurable), rejected_outside_triangle_count=rejected_count, maximum_depth_reached=maximum_depth_reached, stopped_by_depth_limit=bool(depth_limited or unmeasurable))
_EPSILON = 1e-12

@dataclass(frozen=True, slots=True)
class Point2D:
    x: float
    y: float

@dataclass(frozen=True, slots=True)
class BarycentricCoordinates:
    first_weight: float
    second_weight: float
    third_weight: float
    inside_closed_triangle: bool

    @property
    def weight_sum(self) -> float:
        return self.first_weight + self.second_weight + self.third_weight

def _bary_validated_triangle_denominator(triangle: Sequence[Point2D], *, degeneracy_epsilon: float) -> tuple[tuple[Point2D, Point2D, Point2D], float, float]:
    if len(triangle) != 3:
        raise ValueError('a triangle must contain exactly three points')
    epsilon = float(degeneracy_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError('degeneracy_epsilon must be positive and finite')
    vertices = tuple(triangle)
    values = tuple((coordinate for item in vertices for coordinate in (item.x, item.y)))
    if not all((math.isfinite(value) for value in values)):
        raise ValueError('triangle coordinates must be finite')
    first, second, third = vertices
    denominator = (second.y - third.y) * (first.x - third.x) + (third.x - second.x) * (first.y - third.y)
    coordinate_scale = max(1.0, *(abs(value) for value in values))
    area_tolerance = epsilon * coordinate_scale * coordinate_scale
    return (vertices, denominator, area_tolerance)

def triangle_is_degenerate(triangle: Sequence[Point2D], *, degeneracy_epsilon: float=_EPSILON) -> bool:
    """Return whether a finite 2D triangle is degenerate at the given scale."""
    _, denominator, area_tolerance = _bary_validated_triangle_denominator(triangle, degeneracy_epsilon=degeneracy_epsilon)
    return abs(denominator) <= area_tolerance

def triangle_barycentric_coordinates(triangle: Sequence[Point2D], point: Point2D, *, degeneracy_epsilon: float=_EPSILON) -> BarycentricCoordinates:
    """Return affine barycentric weights of a point relative to a 2D triangle."""
    vertices, denominator, area_tolerance = _bary_validated_triangle_denominator(triangle, degeneracy_epsilon=degeneracy_epsilon)
    if not all((math.isfinite(value) for value in (point.x, point.y))):
        raise ValueError('sample-point coordinates must be finite')
    if abs(denominator) <= area_tolerance:
        raise ValueError('triangle must be non-degenerate')
    first, second, third = vertices
    first_weight = ((second.y - third.y) * (point.x - third.x) + (third.x - second.x) * (point.y - third.y)) / denominator
    second_weight = ((third.y - first.y) * (point.x - third.x) + (first.x - third.x) * (point.y - third.y)) / denominator
    third_weight = 1.0 - first_weight - second_weight
    epsilon = float(degeneracy_epsilon)
    weight_tolerance = epsilon * 8.0
    inside = all((-weight_tolerance <= weight <= 1.0 + weight_tolerance for weight in (first_weight, second_weight, third_weight)))
    return BarycentricCoordinates(first_weight=first_weight, second_weight=second_weight, third_weight=third_weight, inside_closed_triangle=inside)
LocationType = Literal['vertex', 'edge', 'interior']
_EPSILON = 1e-12

@dataclass(frozen=True, slots=True)
class TriangleConeIntersection:
    intersects: bool
    minimum_normalized_radius_squared: float
    cone_radius_squared: float
    margin_squared: float
    minimizing_location_type: LocationType
    minimizing_feature_indices: tuple[int, ...]
    closest_normalized_point: tuple[float, float]

def _cone_validate_max_angle(max_angle_rad: float) -> float:
    maximum = float(max_angle_rad)
    if not math.isfinite(maximum) or not 0.0 < maximum < math.pi / 2.0:
        raise ValueError('max_angle_rad must be in (0, pi/2)')
    return maximum

def _cone_normalized_point(point: Vector3) -> tuple[float, float]:
    if not all((math.isfinite(value) for value in (point.x, point.y, point.z))):
        raise ValueError('triangle coordinates must be finite')
    if point.z <= 0.0:
        raise ValueError('triangle vertices must have positive depth')
    return (point.x / point.z, point.y / point.z)

def _cone_cross_2d(first: tuple[float, float], second: tuple[float, float], third: tuple[float, float]) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (third[0] - first[0])

def _cone_origin_in_triangle(points: tuple[tuple[float, float], ...]) -> bool:
    first, second, third = points
    values = (_cone_cross_2d(first, second, (0.0, 0.0)), _cone_cross_2d(second, third, (0.0, 0.0)), _cone_cross_2d(third, first, (0.0, 0.0)))
    has_positive = any((value > _EPSILON for value in values))
    has_negative = any((value < -_EPSILON for value in values))
    on_boundary = any((abs(value) <= _EPSILON for value in values))
    return not (has_positive and has_negative) and (not on_boundary)

def _cone_closest_on_segment(first: tuple[float, float], second: tuple[float, float]) -> tuple[tuple[float, float], float]:
    delta_x = second[0] - first[0]
    delta_y = second[1] - first[1]
    length_squared = delta_x * delta_x + delta_y * delta_y
    if length_squared <= _EPSILON:
        return (first, 0.0)
    ratio = -(first[0] * delta_x + first[1] * delta_y) / length_squared
    ratio = min(1.0, max(0.0, ratio))
    return ((first[0] + ratio * delta_x, first[1] + ratio * delta_y), ratio)

def triangle_intersects_angular_fov_cone(triangle_camera: Sequence[Vector3], *, max_angle_rad: float) -> TriangleConeIntersection:
    """Return exact intersection with the positive angular FOV cone.

    The cone boundary is included. Degenerate normalized triangles are handled
    through their vertices and edges.
    """
    if len(triangle_camera) != 3:
        raise ValueError('a triangle must contain exactly three vertices')
    maximum = _cone_validate_max_angle(max_angle_rad)
    points = tuple((_cone_normalized_point(point) for point in triangle_camera))
    cone_radius_squared = math.tan(maximum) ** 2
    area_twice = abs(_cone_cross_2d(points[0], points[1], points[2]))
    if area_twice > _EPSILON and _cone_origin_in_triangle(points):
        minimum_squared = 0.0
        closest = (0.0, 0.0)
        location_type: LocationType = 'interior'
        indices: tuple[int, ...] = (0, 1, 2)
    else:
        candidates: list[tuple[float, tuple[float, float], LocationType, tuple[int, ...]]] = []
        for index, point in enumerate(points):
            squared = point[0] * point[0] + point[1] * point[1]
            candidates.append((squared, point, 'vertex', (index,)))
        for first_index, second_index in ((0, 1), (1, 2), (2, 0)):
            point, ratio = _cone_closest_on_segment(points[first_index], points[second_index])
            squared = point[0] * point[0] + point[1] * point[1]
            if ratio <= _EPSILON:
                feature_type: LocationType = 'vertex'
                feature_indices = (first_index,)
            elif ratio >= 1.0 - _EPSILON:
                feature_type = 'vertex'
                feature_indices = (second_index,)
            else:
                feature_type = 'edge'
                feature_indices = (first_index, second_index)
            candidates.append((squared, point, feature_type, feature_indices))
        minimum_squared, closest, location_type, indices = min(candidates, key=lambda item: (item[0], len(item[3]), item[3]))
    scale = max(1.0, minimum_squared, cone_radius_squared)
    intersects = minimum_squared <= cone_radius_squared + _EPSILON * scale
    return TriangleConeIntersection(intersects=intersects, minimum_normalized_radius_squared=minimum_squared, cone_radius_squared=cone_radius_squared, margin_squared=minimum_squared - cone_radius_squared, minimizing_location_type=location_type, minimizing_feature_indices=indices, closest_normalized_point=closest)

def _facing_subtract(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(first.x - second.x, first.y - second.y, first.z - second.z)

def _facing_cross(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(first.y * second.z - first.z * second.y, first.z * second.x - first.x * second.z, first.x * second.y - first.y * second.x)

def _facing_dot(first: Vector3, second: Vector3) -> float:
    return first.x * second.x + first.y * second.y + first.z * second.z

def _facing_validate_triangle(triangle: Sequence[Vector3]) -> Triangle3D:
    if len(triangle) != 3:
        raise ValueError('a triangle must contain exactly three vertices')
    vertices = tuple(triangle)
    for vertex in vertices:
        if not all((math.isfinite(value) for value in (vertex.x, vertex.y, vertex.z))):
            raise ValueError('triangle vertices must be finite')
    return vertices

def triangle_normal(triangle: Sequence[Vector3]) -> Vector3:
    """Return the unnormalized winding-derived triangle normal."""
    first, second, third = _facing_validate_triangle(triangle)
    normal = _facing_cross(_facing_subtract(second, first), _facing_subtract(third, first))
    if _facing_dot(normal, normal) <= 1e-24:
        raise ValueError('triangle must have non-zero area')
    return normal

def is_triangle_front_facing(triangle: Sequence[Vector3], *, tolerance: float=1e-12) -> bool:
    """Return whether an outward-wound triangle faces the camera origin.

    Edge-on triangles within the tolerance are not considered front-facing.
    """
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError('tolerance must be finite and non-negative')
    first, second, third = _facing_validate_triangle(triangle)
    normal = triangle_normal((first, second, third))
    centroid = Vector3((first.x + second.x + third.x) / 3.0, (first.y + second.y + third.y) / 3.0, (first.z + second.z + third.z) / 3.0)
    view_vector = Vector3(-centroid.x, -centroid.y, -centroid.z)
    return _facing_dot(normal, view_vector) > tolerance_value

def select_front_facing_triangles(triangles: Iterable[Sequence[Vector3]], *, tolerance: float=1e-12) -> tuple[Triangle3D, ...]:
    """Return front-facing triangles while preserving input order."""
    selected: list[Triangle3D] = []
    for triangle in triangles:
        validated = _facing_validate_triangle(triangle)
        if is_triangle_front_facing(validated, tolerance=tolerance):
            selected.append(validated)
    return tuple(selected)

def _near_finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be finite')
    return converted

def _near_intersection(first: Vector3, second: Vector3, near: float) -> Vector3:
    denominator = second.z - first.z
    if abs(denominator) <= 1e-12:
        raise ValueError('cannot intersect an edge parallel to the near plane')
    ratio = (near - first.z) / denominator
    return Vector3(first.x + ratio * (second.x - first.x), first.y + ratio * (second.y - first.y), near)

def clip_polygon_to_positive_z(vertices: Sequence[Vector3], *, near_plane_m: float=0.001) -> tuple[Vector3, ...]:
    """Clip a winding-ordered polygon against z >= near_plane_m."""
    near = _near_finite(near_plane_m, 'near_plane_m')
    if near <= 0.0:
        raise ValueError('near_plane_m must be positive')
    if len(vertices) < 3:
        raise ValueError('a polygon must contain at least three vertices')
    output: list[Vector3] = []
    previous = vertices[-1]
    previous_inside = previous.z >= near
    for current in vertices:
        current_inside = current.z >= near
        if current_inside:
            if not previous_inside:
                output.append(_near_intersection(previous, current, near))
            output.append(current)
        elif previous_inside:
            output.append(_near_intersection(previous, current, near))
        previous = current
        previous_inside = current_inside
    return tuple(output)

def clip_triangle_to_positive_z(triangle: Sequence[Vector3], *, near_plane_m: float=0.001) -> tuple[tuple[Vector3, Vector3, Vector3], ...]:
    """Clip one triangle and return zero, one, or two winding-preserving triangles."""
    if len(triangle) != 3:
        raise ValueError('a triangle must contain exactly three vertices')
    polygon = clip_polygon_to_positive_z(triangle, near_plane_m=near_plane_m)
    if len(polygon) < 3:
        return ()
    if len(polygon) == 3:
        return ((polygon[0], polygon[1], polygon[2]),)
    if len(polygon) == 4:
        return ((polygon[0], polygon[1], polygon[2]), (polygon[0], polygon[2], polygon[3]))
    raise RuntimeError(f'triangle clipping unexpectedly produced {len(polygon)} vertices')
_DEFAULT_TOLERANCE = 1e-12

@dataclass(frozen=True, slots=True)
class PerspectiveDepthInterpolation:
    reciprocal_depth_per_m: float
    depth_m: float

def interpolate_perspective_camera_depth(vertex_depths_m: Sequence[float], barycentric_weights: Sequence[float], *, weight_tolerance: float=_DEFAULT_TOLERANCE) -> PerspectiveDepthInterpolation:
    """Interpolate positive camera depth from screen-space barycentric weights.

    The weights must describe a point in the closed projected triangle: each
    weight is within [0, 1] up to tolerance and their sum is one up to tolerance.
    """
    if len(vertex_depths_m) != 3:
        raise ValueError('vertex_depths_m must contain exactly three values')
    if len(barycentric_weights) != 3:
        raise ValueError('barycentric_weights must contain exactly three values')
    tolerance = float(weight_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError('weight_tolerance must be positive and finite')
    depths = tuple((float(value) for value in vertex_depths_m))
    weights = tuple((float(value) for value in barycentric_weights))
    if not all((math.isfinite(value) for value in depths)):
        raise ValueError('vertex depths must be finite')
    if not all((value > 0.0 for value in depths)):
        raise ValueError('vertex depths must be positive')
    if not all((math.isfinite(value) for value in weights)):
        raise ValueError('barycentric weights must be finite')
    weight_sum = sum(weights)
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError('barycentric weights must sum to one')
    if not all((-tolerance <= weight <= 1.0 + tolerance for weight in weights)):
        raise ValueError('barycentric weights must describe the closed triangle')
    reciprocal_depth = sum((weight / depth for weight, depth in zip(weights, depths)))
    if not math.isfinite(reciprocal_depth) or reciprocal_depth <= 0.0:
        raise ValueError('interpolated reciprocal depth must be positive and finite')
    depth = 1.0 / reciprocal_depth
    if not math.isfinite(depth) or depth <= 0.0:
        raise ValueError('interpolated depth must be positive and finite')
    return PerspectiveDepthInterpolation(reciprocal_depth_per_m=reciprocal_depth, depth_m=depth)

@dataclass(frozen=True, slots=True)
class BoxTriangle3D:
    face_name: str
    corner_indices: tuple[int, int, int]
    vertices: tuple[Vector3, Vector3, Vector3]
BOX_FACE_INDEX_QUADS: Final[tuple[tuple[str, tuple[int, int, int, int]], ...]] = (('negative_x', (0, 4, 6, 2)), ('positive_x', (1, 3, 7, 5)), ('negative_y', (0, 1, 5, 4)), ('positive_y', (2, 6, 7, 3)), ('negative_z', (0, 2, 3, 1)), ('positive_z', (4, 5, 7, 6)))
BOX_TRIANGLE_INDEX_TRIPLES: Final[tuple[tuple[str, tuple[int, int, int]], ...]] = tuple((triangle for face_name, (first, second, third, fourth) in BOX_FACE_INDEX_QUADS for triangle in ((face_name, (first, second, third)), (face_name, (first, third, fourth)))))

def triangulate_box_surfaces(corners: Sequence[Vector3]) -> tuple[BoxTriangle3D, ...]:
    """Return the twelve outward-wound triangles of an eight-corner box."""
    if len(corners) != 8:
        raise ValueError('an Actor box must contain exactly eight corners')
    return tuple((BoxTriangle3D(face_name=face_name, corner_indices=indices, vertices=tuple((corners[index] for index in indices))) for face_name, indices in BOX_TRIANGLE_INDEX_TRIPLES))

@dataclass(frozen=True, slots=True)
class CameraFacingBoxTriangle:
    face_name: str
    source_corner_indices: tuple[int, int, int]
    vertices_camera: tuple[Vector3, Vector3, Vector3]

def prepare_camera_facing_box_triangles(corners_camera: Sequence[Vector3], *, near_plane_m: float=0.001, facing_tolerance: float=1e-12) -> tuple[CameraFacingBoxTriangle, ...]:
    """Return near-clipped triangles from camera-facing Actor-box surfaces.

    Source-face visibility is determined before near-plane clipping. A source
    triangle can therefore produce zero, one, or two output triangles.
    Output order follows the fixed box topology and then clipping fan order.
    """
    near = float(near_plane_m)
    tolerance = float(facing_tolerance)
    if not math.isfinite(near) or near <= 0.0:
        raise ValueError('near_plane_m must be finite and positive')
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('facing_tolerance must be finite and non-negative')
    prepared: list[CameraFacingBoxTriangle] = []
    for source in triangulate_box_surfaces(corners_camera):
        if not is_triangle_front_facing(source.vertices, tolerance=tolerance):
            continue
        clipped = clip_triangle_to_positive_z(source.vertices, near_plane_m=near)
        for triangle in clipped:
            prepared.append(CameraFacingBoxTriangle(face_name=source.face_name, source_corner_indices=source.corner_indices, vertices_camera=triangle))
    return tuple(prepared)
