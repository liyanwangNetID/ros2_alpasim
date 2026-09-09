#!/usr/bin/env python3
"""Conservative recursive angular-FOV classification of camera triangles.

This module approximates a triangle/cone intersection by recursive 1-to-4
subdivision. It never silently accepts or rejects an unresolved boundary leaf.
It does not project pixels, clip against the image, rasterize, or compare depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from camera_projection_v01 import Vector3
from triangle_angular_fov_diagnostics_v01 import (
    classify_triangle_angular_fov,
    point_is_within_angular_fov,
)
from triangle_cone_intersection_v01 import (
    triangle_intersects_angular_fov_cone,
)
from ftheta_fov_boundary_projected_extent_v01 import (
    summarize_angular_fov_boundary_projected_extent,
)

Triangle3D = tuple[Vector3, Vector3, Vector3]


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
        # Backward-compatible union of unresolved boundary outputs.
        return (
            self.boundary_depth_limited_triangles
            + self.boundary_unmeasurable_triangles
        )


def _finite_point(point: Vector3) -> None:
    if not all(math.isfinite(value) for value in (point.x, point.y, point.z)):
        raise ValueError("triangle coordinates must be finite")
    if point.z <= 0.0:
        raise ValueError("triangle vertices must have positive depth")


def _midpoint(first: Vector3, second: Vector3) -> Vector3:
    return Vector3(
        0.5 * (first.x + second.x),
        0.5 * (first.y + second.y),
        0.5 * (first.z + second.z),
    )


def _centroid(first: Vector3, second: Vector3, third: Vector3) -> Vector3:
    return Vector3(
        (first.x + second.x + third.x) / 3.0,
        (first.y + second.y + third.y) / 3.0,
        (first.z + second.z + third.z) / 3.0,
    )


def subdivide_triangle_to_angular_fov(
    triangle_camera: Sequence[Vector3],
    *,
    max_angle_rad: float | None,
    maximum_depth: int,
    calibration=None,
    maximum_boundary_extent_px: float | None = None,
) -> AngularFovTriangleSubdivision:
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
        raise ValueError("a triangle must contain exactly three vertices")
    if isinstance(maximum_depth, bool) or not isinstance(maximum_depth, int):
        raise TypeError("maximum_depth must be an integer")
    if maximum_depth < 0:
        raise ValueError("maximum_depth must be non-negative")
    if maximum_boundary_extent_px is not None:
        boundary_limit = float(maximum_boundary_extent_px)
        if not math.isfinite(boundary_limit) or boundary_limit <= 0.0:
            raise ValueError(
                "maximum_boundary_extent_px must be positive and finite"
            )
        if calibration is None:
            raise ValueError(
                "calibration is required when maximum_boundary_extent_px is set"
            )
    else:
        boundary_limit = None

    root: Triangle3D = tuple(triangle_camera)  # type: ignore[assignment]
    for point in root:
        _finite_point(point)

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
            intersection = triangle_intersects_angular_fov_cone(
                vertices,
                max_angle_rad=max_angle_rad,
            )
            if not intersection.intersects:
                rejected_count += 1
                return

        first, second, third = vertices
        midpoint_ab = _midpoint(first, second)
        midpoint_bc = _midpoint(second, third)
        midpoint_ca = _midpoint(third, first)
        centroid = _centroid(first, second, third)
        samples = (
            first, second, third,
            midpoint_ab, midpoint_bc, midpoint_ca,
            centroid,
        )
        sample_inside = tuple(
            point_is_within_angular_fov(point, max_angle_rad=max_angle_rad)
            for point in samples
        )
        inside_count = sum(sample_inside)

        if inside_count == len(samples):
            accepted.append(
                AngularFovAcceptedTriangle(
                    vertices_camera=vertices,
                    subdivision_depth=depth,
                )
            )
            return

        diagnostics = classify_triangle_angular_fov(
            vertices,
            max_angle_rad=max_angle_rad,
        )
        edge_intersection_count = sum(
            edge.clipped_segment is not None
            for edge in diagnostics.edge_intersections
        )

        boundary_extent = None
        measurable = False
        if boundary_limit is not None:
            extent = summarize_angular_fov_boundary_projected_extent(
                vertices,
                calibration,
                max_angle_rad=max_angle_rad,
            )
            measurable = extent.has_measurable_extent
            boundary_extent = extent.maximum_pair_distance_px
            if (
                measurable
                and boundary_extent is not None
                and boundary_extent <= boundary_limit
            ):
                approximated.append(
                    AngularFovBoundaryTriangle(
                        vertices_camera=vertices,
                        subdivision_depth=depth,
                        sample_inside_count=inside_count,
                        edge_intersection_count=edge_intersection_count,
                        maximum_boundary_extent_px=boundary_extent,
                    )
                )
                return

        if depth >= maximum_depth:
            boundary = AngularFovBoundaryTriangle(
                vertices_camera=vertices,
                subdivision_depth=depth,
                sample_inside_count=inside_count,
                edge_intersection_count=edge_intersection_count,
                maximum_boundary_extent_px=boundary_extent,
            )
            if boundary_limit is not None and not measurable:
                unmeasurable.append(boundary)
            else:
                depth_limited.append(boundary)
            return

        children: tuple[Triangle3D, ...] = (
            (first, midpoint_ab, midpoint_ca),
            (midpoint_ab, second, midpoint_bc),
            (midpoint_ca, midpoint_bc, third),
            (midpoint_ab, midpoint_bc, midpoint_ca),
        )
        for child in children:
            recurse(child, depth + 1)

    recurse(root, 0)
    return AngularFovTriangleSubdivision(
        accepted_inside_triangles=tuple(accepted),
        boundary_approximated_triangles=tuple(approximated),
        boundary_depth_limited_triangles=tuple(depth_limited),
        boundary_unmeasurable_triangles=tuple(unmeasurable),
        rejected_outside_triangle_count=rejected_count,
        maximum_depth_reached=maximum_depth_reached,
        stopped_by_depth_limit=bool(depth_limited or unmeasurable),
    )
