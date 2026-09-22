"""Step 7 projected-triangle raster, surface-depth, and Actor z-buffer domain."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence
from step7.geometry import Point2D, triangle_barycentric_coordinates, triangle_is_degenerate
from step7.geometry import interpolate_perspective_camera_depth
from step7.projection import Vector3
from step7.geometry import build_safe_angular_fov_polygon
from step7.geometry import AngularFovTriangleSubdivision, subdivide_triangle_to_angular_fov
from typing import Protocol, Sequence
from step7.projection import PixelProjection, Vector3

@dataclass(frozen=True, slots=True)
class PixelPoint:
    u: float
    v: float

@dataclass(frozen=True, slots=True)
class RasterCell:
    column: int
    row: int

@dataclass(frozen=True, slots=True)
class RasterizedTriangleCoverage:
    cells: tuple[RasterCell, ...]
    candidate_column_range: tuple[int, int] | None
    candidate_row_range: tuple[int, int] | None
_EPSILON = 1e-12

def _cells_cross(first: PixelPoint, second: PixelPoint, third: PixelPoint) -> float:
    return (second.u - first.u) * (third.v - first.v) - (second.v - first.v) * (third.u - first.u)

def _cells_point_in_closed_triangle(point: PixelPoint, triangle: tuple[PixelPoint, ...]) -> bool:
    first, second, third = triangle
    values = (_cells_cross(first, second, point), _cells_cross(second, third, point), _cells_cross(third, first, point))
    return not (any((value > _EPSILON for value in values)) and any((value < -_EPSILON for value in values)))

def _cells_orientation(first: PixelPoint, second: PixelPoint, third: PixelPoint) -> int:
    value = _cells_cross(first, second, third)
    if value > _EPSILON:
        return 1
    if value < -_EPSILON:
        return -1
    return 0

def _cells_on_segment(first: PixelPoint, second: PixelPoint, point: PixelPoint) -> bool:
    return min(first.u, second.u) - _EPSILON <= point.u <= max(first.u, second.u) + _EPSILON and min(first.v, second.v) - _EPSILON <= point.v <= max(first.v, second.v) + _EPSILON and (_cells_orientation(first, second, point) == 0)

def _cells_closed_segments_intersect(first_a: PixelPoint, first_b: PixelPoint, second_a: PixelPoint, second_b: PixelPoint) -> bool:
    o1 = _cells_orientation(first_a, first_b, second_a)
    o2 = _cells_orientation(first_a, first_b, second_b)
    o3 = _cells_orientation(second_a, second_b, first_a)
    o4 = _cells_orientation(second_a, second_b, first_b)
    if o1 != o2 and o3 != o4:
        return True
    return o1 == 0 and _cells_on_segment(first_a, first_b, second_a) or (o2 == 0 and _cells_on_segment(first_a, first_b, second_b)) or (o3 == 0 and _cells_on_segment(second_a, second_b, first_a)) or (o4 == 0 and _cells_on_segment(second_a, second_b, first_b))

def _cells_cell_intersects_triangle(column: int, row: int, triangle: tuple[PixelPoint, ...]) -> bool:
    corners = (PixelPoint(float(column), float(row)), PixelPoint(float(column + 1), float(row)), PixelPoint(float(column + 1), float(row + 1)), PixelPoint(float(column), float(row + 1)))
    if any((_cells_point_in_closed_triangle(corner, triangle) for corner in corners)):
        return True
    if any((column - _EPSILON <= vertex.u <= column + 1 + _EPSILON and row - _EPSILON <= vertex.v <= row + 1 + _EPSILON for vertex in triangle)):
        return True
    triangle_edges = ((0, 1), (1, 2), (2, 0))
    cell_edges = ((0, 1), (1, 2), (2, 3), (3, 0))
    return any((_cells_closed_segments_intersect(triangle[first_triangle], triangle[second_triangle], corners[first_cell], corners[second_cell]) for first_triangle, second_triangle in triangle_edges for first_cell, second_cell in cell_edges))

def rasterize_projected_triangle_cells(triangle_pixels: Sequence[PixelPoint], *, image_width_px: int, image_height_px: int, raster_width: int, raster_height: int) -> RasterizedTriangleCoverage:
    """Return raster cells whose closed area intersects the projected triangle."""
    if len(triangle_pixels) != 3:
        raise ValueError('a triangle must contain exactly three pixel points')
    dimensions = (image_width_px, image_height_px, raster_width, raster_height)
    if any((isinstance(value, bool) or not isinstance(value, int) for value in dimensions)):
        raise TypeError('image and raster dimensions must be integers')
    if any((value <= 0 for value in dimensions)):
        raise ValueError('image and raster dimensions must be positive')
    triangle_source = tuple(triangle_pixels)
    for point in triangle_source:
        if not math.isfinite(point.u) or not math.isfinite(point.v):
            raise ValueError('triangle pixel coordinates must be finite')
    scale_u = raster_width / image_width_px
    scale_v = raster_height / image_height_px
    triangle = tuple((PixelPoint(point.u * scale_u, point.v * scale_v) for point in triangle_source))
    minimum_u = min((point.u for point in triangle))
    minimum_v = min((point.v for point in triangle))
    maximum_u = max((point.u for point in triangle))
    maximum_v = max((point.v for point in triangle))
    min_u = max(0, math.ceil(minimum_u) - 1)
    min_v = max(0, math.ceil(minimum_v) - 1)
    max_u = min(raster_width - 1, math.floor(maximum_u))
    max_v = min(raster_height - 1, math.floor(maximum_v))
    if max_u < min_u or max_v < min_v:
        return RasterizedTriangleCoverage((), None, None)
    cells = tuple((RasterCell(column=column, row=row) for row in range(min_v, max_v + 1) for column in range(min_u, max_u + 1) if _cells_cell_intersects_triangle(column, row, triangle)))
    return RasterizedTriangleCoverage(cells=cells, candidate_column_range=(min_u, max_u), candidate_row_range=(min_v, max_v))

@dataclass(frozen=True, slots=True)
class RasterDepthSample:
    cell: RasterCell
    sample_u_px: float
    sample_v_px: float
    barycentric_weights: tuple[float, float, float]
    reciprocal_depth_per_m: float
    depth_m: float

@dataclass(frozen=True, slots=True)
class ProjectedTriangleDepthSamples:
    conservative_coverage_cells: tuple[RasterCell, ...]
    center_sampled_depths: tuple[RasterDepthSample, ...]
    conservative_cell_count: int
    center_sampled_cell_count: int

def sample_projected_triangle_depths(triangle_pixels: Sequence[PixelPoint], vertex_depths_m: Sequence[float], *, image_width_px: int, image_height_px: int, raster_width: int, raster_height: int) -> ProjectedTriangleDepthSamples:
    """Return center samples and perspective-correct depths for one triangle."""
    if len(triangle_pixels) != 3:
        raise ValueError('a triangle must contain exactly three pixel points')
    if len(vertex_depths_m) != 3:
        raise ValueError('vertex_depths_m must contain exactly three values')
    pixels = tuple(triangle_pixels)
    depths = tuple((float(value) for value in vertex_depths_m))
    if not all((math.isfinite(value) for value in depths)):
        raise ValueError('vertex depths must be finite')
    if not all((value > 0.0 for value in depths)):
        raise ValueError('vertex depths must be positive')
    coverage = rasterize_projected_triangle_cells(pixels, image_width_px=image_width_px, image_height_px=image_height_px, raster_width=raster_width, raster_height=raster_height)
    barycentric_triangle = tuple((Point2D(point.u, point.v) for point in pixels))
    if triangle_is_degenerate(barycentric_triangle):
        return ProjectedTriangleDepthSamples(conservative_coverage_cells=coverage.cells, center_sampled_depths=(), conservative_cell_count=len(coverage.cells), center_sampled_cell_count=0)
    pixel_width_per_cell = image_width_px / raster_width
    pixel_height_per_cell = image_height_px / raster_height
    samples: list[RasterDepthSample] = []
    for cell in coverage.cells:
        sample_u = (cell.column + 0.5) * pixel_width_per_cell
        sample_v = (cell.row + 0.5) * pixel_height_per_cell
        barycentric = triangle_barycentric_coordinates(barycentric_triangle, Point2D(sample_u, sample_v))
        if not barycentric.inside_closed_triangle:
            continue
        weights = (barycentric.first_weight, barycentric.second_weight, barycentric.third_weight)
        interpolation = interpolate_perspective_camera_depth(depths, weights)
        samples.append(RasterDepthSample(cell=cell, sample_u_px=sample_u, sample_v_px=sample_v, barycentric_weights=weights, reciprocal_depth_per_m=interpolation.reciprocal_depth_per_m, depth_m=interpolation.depth_m))
    result_samples = tuple(samples)
    return ProjectedTriangleDepthSamples(conservative_coverage_cells=coverage.cells, center_sampled_depths=result_samples, conservative_cell_count=len(coverage.cells), center_sampled_cell_count=len(result_samples))

@dataclass(frozen=True, slots=True)
class FovSubdividedTriangleDepthSamples:
    subdivision: AngularFovTriangleSubdivision
    accepted_triangle_samples: tuple[ProjectedTriangleDepthSamples, ...]
    boundary_triangle_samples: tuple[ProjectedTriangleDepthSamples, ...]
    all_triangle_samples: tuple[ProjectedTriangleDepthSamples, ...]
    sampled_inside_triangle_count: int
    sampled_boundary_triangle_count: int
    measurable_boundary_polygon_count: int
    unmeasurable_boundary_polygon_count: int
    degenerate_boundary_polygon_count: int
    unresolved_boundary_count: int

def sample_fov_subdivided_camera_triangle_depths(triangle_camera: Sequence[Vector3], calibration, *, max_angle_rad: float | None, maximum_depth: int, maximum_boundary_extent_px: float, image_width_px: int, image_height_px: int, raster_width: int, raster_height: int, near_plane_m: float=0.001) -> FovSubdividedTriangleDepthSamples:
    """Subdivide by angular FOV and sample safe inside geometry."""
    subdivision = subdivide_triangle_to_angular_fov(triangle_camera, max_angle_rad=max_angle_rad, maximum_depth=maximum_depth, calibration=calibration, maximum_boundary_extent_px=maximum_boundary_extent_px)

    def sample_triangle(vertices):
        return sample_camera_triangle_depths(vertices, calibration, image_width_px=image_width_px, image_height_px=image_height_px, raster_width=raster_width, raster_height=raster_height, near_plane_m=near_plane_m)
    accepted_samples = tuple((sample_triangle(accepted.vertices_camera) for accepted in subdivision.accepted_inside_triangles))
    boundary_samples: list[ProjectedTriangleDepthSamples] = []
    measurable_count = 0
    unmeasurable_count = 0
    degenerate_count = 0
    for boundary in subdivision.boundary_approximated_triangles:
        polygon = build_safe_angular_fov_polygon(boundary.vertices_camera, max_angle_rad=max_angle_rad)
        if not polygon.has_measurable_polygon:
            unmeasurable_count += 1
            continue
        if polygon.is_degenerate:
            degenerate_count += 1
            continue
        measurable_count += 1
        boundary_samples.extend((sample_triangle(vertices) for vertices in polygon.triangles_camera))
    boundary_samples_tuple = tuple(boundary_samples)
    all_samples = accepted_samples + boundary_samples_tuple
    return FovSubdividedTriangleDepthSamples(subdivision=subdivision, accepted_triangle_samples=accepted_samples, boundary_triangle_samples=boundary_samples_tuple, all_triangle_samples=all_samples, sampled_inside_triangle_count=len(accepted_samples), sampled_boundary_triangle_count=len(boundary_samples_tuple), measurable_boundary_polygon_count=measurable_count, unmeasurable_boundary_polygon_count=unmeasurable_count, degenerate_boundary_polygon_count=degenerate_count, unresolved_boundary_count=len(subdivision.boundary_depth_limited_triangles) + len(subdivision.boundary_unmeasurable_triangles) + unmeasurable_count + degenerate_count)

class CameraPointProjector(Protocol):

    def project_camera_point(self, point: Vector3) -> PixelProjection:
        ...

def sample_camera_triangle_depths(triangle_camera: Sequence[Vector3], calibration: CameraPointProjector, *, image_width_px: int, image_height_px: int, raster_width: int, raster_height: int, near_plane_m: float=0.001) -> ProjectedTriangleDepthSamples:
    """Project one in-FOV camera triangle and sample its raster-center depths."""
    if len(triangle_camera) != 3:
        raise ValueError('a triangle must contain exactly three camera points')
    near = float(near_plane_m)
    if not math.isfinite(near) or near <= 0.0:
        raise ValueError('near_plane_m must be positive and finite')
    vertices = tuple(triangle_camera)
    for vertex in vertices:
        if not all((math.isfinite(value) for value in (vertex.x, vertex.y, vertex.z))):
            raise ValueError('camera triangle coordinates must be finite')
        if vertex.z < near:
            raise ValueError('camera triangle vertex is behind near_plane_m')
    projections = tuple((calibration.project_camera_point(vertex) for vertex in vertices))
    for projection in projections:
        if not projection.positive_z:
            raise ValueError('projected triangle vertex must have positive depth')
        if not projection.within_fov:
            raise ValueError('projected triangle vertex must be within FOV')
        if not math.isfinite(projection.u) or not math.isfinite(projection.v):
            raise ValueError('projected pixel coordinates must be finite')
    return sample_projected_triangle_depths(tuple((PixelPoint(item.u, item.v) for item in projections)), tuple((vertex.z for vertex in vertices)), image_width_px=image_width_px, image_height_px=image_height_px, raster_width=raster_width, raster_height=raster_height)

@dataclass(frozen=True, slots=True)
class ActorSurfaceCellDepth:
    cell: RasterCell
    depth_m: float
    reciprocal_depth_per_m: float
    winning_triangle_index: int
    sample_u_px: float
    sample_v_px: float
    barycentric_weights: tuple[float, float, float]

@dataclass(frozen=True, slots=True)
class ActorSurfaceDepthRaster:
    cell_depths: tuple[ActorSurfaceCellDepth, ...]
    occupied_cell_count: int
    input_triangle_count: int
    input_depth_sample_count: int
    replaced_sample_count: int
    discarded_farther_or_equal_sample_count: int

def merge_actor_surface_depth_samples(triangle_samples: Sequence[ProjectedTriangleDepthSamples], *, depth_tolerance_m: float=1e-09) -> ActorSurfaceDepthRaster:
    """Keep the nearest triangle sample in each cell for one Actor.

    A new sample replaces the current winner only when it is nearer by more
    than depth_tolerance_m. Equal or tolerance-equivalent samples preserve the
    earlier triangle index, giving deterministic input-order tie handling.
    """
    tolerance = float(depth_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('depth_tolerance_m must be non-negative and finite')
    winners: dict[RasterCell, ActorSurfaceCellDepth] = {}
    input_sample_count = 0
    replaced_count = 0
    discarded_count = 0
    for triangle_index, result in enumerate(triangle_samples):
        for sample in result.center_sampled_depths:
            input_sample_count += 1
            values = (sample.depth_m, sample.reciprocal_depth_per_m, sample.sample_u_px, sample.sample_v_px, *sample.barycentric_weights)
            if not all((math.isfinite(value) for value in values)):
                raise ValueError('triangle depth samples must be finite')
            if sample.depth_m <= 0.0 or sample.reciprocal_depth_per_m <= 0.0:
                raise ValueError('triangle depth samples must have positive depth')
            if not math.isclose(sample.depth_m * sample.reciprocal_depth_per_m, 1.0, rel_tol=1e-09, abs_tol=1e-12):
                raise ValueError('depth and reciprocal depth must be consistent')
            candidate = ActorSurfaceCellDepth(cell=sample.cell, depth_m=sample.depth_m, reciprocal_depth_per_m=sample.reciprocal_depth_per_m, winning_triangle_index=triangle_index, sample_u_px=sample.sample_u_px, sample_v_px=sample.sample_v_px, barycentric_weights=sample.barycentric_weights)
            current = winners.get(sample.cell)
            if current is None:
                winners[sample.cell] = candidate
            elif candidate.depth_m < current.depth_m - tolerance:
                winners[sample.cell] = candidate
                replaced_count += 1
            else:
                discarded_count += 1
    ordered = tuple((winners[cell] for cell in sorted(winners, key=lambda item: (item.row, item.column))))
    return ActorSurfaceDepthRaster(cell_depths=ordered, occupied_cell_count=len(ordered), input_triangle_count=len(triangle_samples), input_depth_sample_count=input_sample_count, replaced_sample_count=replaced_count, discarded_farther_or_equal_sample_count=discarded_count)
Triangle3D = tuple[Vector3, Vector3, Vector3]

@dataclass(frozen=True, slots=True)
class ActorCameraSurfaceDepthRaster:
    surface_raster: ActorSurfaceDepthRaster
    triangle_results: tuple[FovSubdividedTriangleDepthSamples, ...]
    source_triangle_count: int
    generated_triangle_sample_count: int
    unresolved_boundary_count: int
    zero_center_sample_triangle_count: int

def build_actor_camera_surface_depth_raster(triangles_camera: Sequence[Sequence[Vector3]], calibration, *, max_angle_rad: float | None, maximum_depth: int, maximum_boundary_extent_px: float, image_width_px: int, image_height_px: int, raster_width: int, raster_height: int, near_plane_m: float=0.001, depth_tolerance_m: float=1e-09) -> ActorCameraSurfaceDepthRaster:
    """Process and merge all prepared camera-frame triangles for one Actor."""
    source = tuple((tuple(triangle) for triangle in triangles_camera))
    results = tuple((sample_fov_subdivided_camera_triangle_depths(triangle, calibration, max_angle_rad=max_angle_rad, maximum_depth=maximum_depth, maximum_boundary_extent_px=maximum_boundary_extent_px, image_width_px=image_width_px, image_height_px=image_height_px, raster_width=raster_width, raster_height=raster_height, near_plane_m=near_plane_m) for triangle in source))
    generated_samples = tuple((sample for result in results for sample in result.all_triangle_samples))
    raster = merge_actor_surface_depth_samples(generated_samples, depth_tolerance_m=depth_tolerance_m)
    return ActorCameraSurfaceDepthRaster(surface_raster=raster, triangle_results=results, source_triangle_count=len(source), generated_triangle_sample_count=len(generated_samples), unresolved_boundary_count=sum((result.unresolved_boundary_count for result in results)), zero_center_sample_triangle_count=sum((sample.center_sampled_cell_count == 0 for sample in generated_samples)))

@dataclass(frozen=True, slots=True)
class ActorDepthRasterInput:
    actor_id: str
    surface_raster: ActorSurfaceDepthRaster

@dataclass(frozen=True, slots=True)
class ZBufferCellWinner:
    cell: RasterCell
    actor_id: str
    depth_m: float
    source_surface: ActorSurfaceCellDepth

@dataclass(frozen=True, slots=True)
class ActorZBufferSummary:
    actor_id: str
    occupied_cell_count: int
    winning_cell_count: int
    occluded_cell_count: int

@dataclass(frozen=True, slots=True)
class ActorDepthZBuffer:
    cell_winners: tuple[ZBufferCellWinner, ...]
    actor_summaries: tuple[ActorZBufferSummary, ...]
    occupied_union_cell_count: int
    contested_cell_count: int

def resolve_actor_depth_zbuffer(actor_rasters: Sequence[ActorDepthRasterInput], *, depth_tolerance_m: float=1e-09) -> ActorDepthZBuffer:
    """Resolve nearest Actor per cell with order-independent tie handling.

    A candidate wins if it is nearer by more than depth_tolerance_m. Depths
    within tolerance are tied and the lexicographically smaller actor_id wins.
    Actor IDs must be non-empty and unique.
    """
    tolerance = float(depth_tolerance_m)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError('depth_tolerance_m must be non-negative and finite')
    inputs = tuple(actor_rasters)
    actor_ids = tuple((item.actor_id for item in inputs))
    if any((not isinstance(actor_id, str) or not actor_id for actor_id in actor_ids)):
        raise ValueError('actor_id must be a non-empty string')
    if len(set(actor_ids)) != len(actor_ids):
        raise ValueError('actor_id values must be unique')
    candidates_by_cell: dict[RasterCell, list[tuple[str, ActorSurfaceCellDepth]]] = {}
    occupied_counts: dict[str, int] = {}
    for actor_input in inputs:
        depths = actor_input.surface_raster.cell_depths
        if actor_input.surface_raster.occupied_cell_count != len(depths):
            raise ValueError('surface raster occupied_cell_count is inconsistent')
        seen_cells: set[RasterCell] = set()
        for surface in depths:
            if surface.cell in seen_cells:
                raise ValueError('surface raster contains duplicate cells')
            seen_cells.add(surface.cell)
            if not math.isfinite(surface.depth_m) or surface.depth_m <= 0.0:
                raise ValueError('surface depths must be positive and finite')
            candidates_by_cell.setdefault(surface.cell, []).append((actor_input.actor_id, surface))
        occupied_counts[actor_input.actor_id] = len(depths)
    winners: list[ZBufferCellWinner] = []
    winning_counts = {actor_id: 0 for actor_id in actor_ids}
    contested_count = 0
    for cell in sorted(candidates_by_cell, key=lambda item: (item.row, item.column)):
        candidates = candidates_by_cell[cell]
        if len(candidates) > 1:
            contested_count += 1
        minimum_depth = min((surface.depth_m for _, surface in candidates))
        tied = [(actor_id, surface) for actor_id, surface in candidates if surface.depth_m <= minimum_depth + tolerance]
        winner_actor_id, winner_surface = min(tied, key=lambda item: item[0])
        winners.append(ZBufferCellWinner(cell=cell, actor_id=winner_actor_id, depth_m=winner_surface.depth_m, source_surface=winner_surface))
        winning_counts[winner_actor_id] += 1
    summaries = tuple((ActorZBufferSummary(actor_id=actor_id, occupied_cell_count=occupied_counts[actor_id], winning_cell_count=winning_counts[actor_id], occluded_cell_count=occupied_counts[actor_id] - winning_counts[actor_id]) for actor_id in sorted(actor_ids)))
    return ActorDepthZBuffer(cell_winners=tuple(winners), actor_summaries=summaries, occupied_union_cell_count=len(winners), contested_cell_count=contested_count)
