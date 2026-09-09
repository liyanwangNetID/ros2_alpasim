#!/usr/bin/env python3
"""Step 7D.4 Actor 3D box projection into one raw F-theta camera image.

This module produces continuous projection geometry only. It does not apply
observability thresholds, distance thresholds, role selection, or occlusion
classification.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from adaptive_ftheta_sampling_v01 import sample_projected_edge_adaptive
from actor_box_projection_v01 import actor_box_corners_in_rig
from camera_projection_v01 import (
    FthetaCameraCalibration,
    PixelProjection,
    Vector3,
)
from ftheta_fov_clipping_v01 import clip_segment_to_angular_fov
from projected_hull_v01 import Point2D, summarize_projected_hull

# Corner ordering produced by actor_box_projection_v01.local_box_corners():
# z varies slowest, then y, then x.
BOX_EDGE_INDEX_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@dataclass(frozen=True, slots=True)
class BoundingBox2D:
    min_u: float
    min_v: float
    max_u: float
    max_v: float

    @property
    def width(self) -> float:
        return max(0.0, self.max_u - self.min_u)

    @property
    def height(self) -> float:
        return max(0.0, self.max_v - self.min_v)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActorCameraProjection:
    camera_name: str
    track_id: str
    actor_class: str
    corner_count: int
    edge_count: int
    edge_samples_per_edge: int
    camera_sample_count: int
    positive_depth_sample_count: int
    within_fov_sample_count: int
    inside_image_sample_count: int
    projected_bbox: BoundingBox2D | None
    clipped_bbox: BoundingBox2D | None
    projected_area_px: float
    inside_image_area_px: float
    inside_image_ratio: float
    projected_hull: tuple[Point2D, ...]
    clipped_hull: tuple[Point2D, ...]
    projected_hull_area_px: float
    inside_image_hull_area_px: float
    inside_image_hull_ratio: float
    projected_height_px: float
    minimum_depth_m: float | None
    maximum_depth_m: float | None
    truncated: bool
    projection_valid: bool
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        return value


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _interpolate(first: Vector3, second: Vector3, ratio: float) -> Vector3:
    return Vector3(
        first.x + ratio * (second.x - first.x),
        first.y + ratio * (second.y - first.y),
        first.z + ratio * (second.z - first.z),
    )


def clip_segment_to_positive_z(
    first: Vector3,
    second: Vector3,
    *,
    near_plane_m: float,
) -> tuple[Vector3, Vector3] | None:
    """Clip a camera-frame segment to z >= near_plane_m."""
    near = _finite(near_plane_m, "near_plane_m")
    if near <= 0.0:
        raise ValueError("near_plane_m must be positive")

    first_inside = first.z >= near
    second_inside = second.z >= near
    if first_inside and second_inside:
        return first, second
    if not first_inside and not second_inside:
        return None

    denominator = second.z - first.z
    if abs(denominator) <= 1e-12:
        return None
    ratio = (near - first.z) / denominator
    intersection = _interpolate(first, second, ratio)
    intersection = Vector3(intersection.x, intersection.y, near)
    if first_inside:
        return first, intersection
    return intersection, second


def sample_box_edges_camera(
    corners_camera: Sequence[Vector3],
    *,
    samples_per_edge: int = 9,
    near_plane_m: float = 1e-3,
) -> tuple[Vector3, ...]:
    """Clip and sample all twelve box edges in the camera frame."""
    if len(corners_camera) != 8:
        raise ValueError("an Actor box must contain exactly eight corners")
    if isinstance(samples_per_edge, bool) or not isinstance(samples_per_edge, int):
        raise TypeError("samples_per_edge must be an integer")
    if samples_per_edge < 2:
        raise ValueError("samples_per_edge must be at least two")

    samples: list[Vector3] = []
    for first_index, second_index in BOX_EDGE_INDEX_PAIRS:
        clipped = clip_segment_to_positive_z(
            corners_camera[first_index],
            corners_camera[second_index],
            near_plane_m=near_plane_m,
        )
        if clipped is None:
            continue
        first, second = clipped
        for index in range(samples_per_edge):
            ratio = index / (samples_per_edge - 1)
            samples.append(_interpolate(first, second, ratio))
    return tuple(samples)



def sample_box_edges_projected_adaptive(
    corners_camera: Sequence[Vector3],
    *,
    calibration: FthetaCameraCalibration,
    near_plane_m: float = 1e-3,
    maximum_chord_error_px: float = 1.0,
    maximum_adaptive_depth: int = 14,
) -> tuple[tuple[Vector3, ...], tuple[PixelProjection, ...], bool]:
    camera_points: list[Vector3] = []
    projections: list[PixelProjection] = []
    depth_limited = False
    for first_index, second_index in BOX_EDGE_INDEX_PAIRS:
        segment = clip_segment_to_positive_z(
            corners_camera[first_index], corners_camera[second_index],
            near_plane_m=near_plane_m,
        )
        if segment is None:
            continue
        segment = clip_segment_to_angular_fov(
            segment[0], segment[1], max_angle_rad=calibration.max_angle_rad,
        )
        if segment is None:
            continue
        sampled = sample_projected_edge_adaptive(
            segment[0], segment[1], calibration,
            maximum_chord_error_px=maximum_chord_error_px,
            maximum_depth=maximum_adaptive_depth,
        )
        camera_points.extend(sampled.camera_points)
        projections.extend(sampled.projections)
        depth_limited |= sampled.stopped_by_depth_limit
    return tuple(camera_points), tuple(projections), depth_limited


def _bbox(points: Sequence[tuple[float, float]]) -> BoundingBox2D | None:
    if not points:
        return None
    return BoundingBox2D(
        min_u=min(point[0] for point in points),
        min_v=min(point[1] for point in points),
        max_u=max(point[0] for point in points),
        max_v=max(point[1] for point in points),
    )


def _clip_bbox_to_image(
    bbox: BoundingBox2D,
    *,
    width: int,
    height: int,
) -> BoundingBox2D | None:
    clipped = BoundingBox2D(
        min_u=max(0.0, bbox.min_u),
        min_v=max(0.0, bbox.min_v),
        max_u=min(float(width - 1), bbox.max_u),
        max_v=min(float(height - 1), bbox.max_v),
    )
    if clipped.max_u <= clipped.min_u or clipped.max_v <= clipped.min_v:
        return None
    return clipped


def summarize_camera_box_projection(
    corners_camera: Sequence[Vector3],
    *,
    calibration: FthetaCameraCalibration,
    track_id: str,
    actor_class: str,
    samples_per_edge: int | None = None,
    near_plane_m: float = 1e-3,
    maximum_chord_error_px: float = 1.0,
    maximum_adaptive_depth: int = 14,
) -> ActorCameraProjection:
    """Project a camera-frame 3D box and summarize its 2D geometry."""
    if samples_per_edge is None:
        samples, projections, depth_limited = sample_box_edges_projected_adaptive(
            corners_camera,
            calibration=calibration,
            near_plane_m=near_plane_m,
            maximum_chord_error_px=maximum_chord_error_px,
            maximum_adaptive_depth=maximum_adaptive_depth,
        )
        reported_samples_per_edge = 0
    else:
        samples = sample_box_edges_camera(
            corners_camera,
            samples_per_edge=samples_per_edge,
            near_plane_m=near_plane_m,
        )
        projections = tuple(
            calibration.project_camera_point(point) for point in samples
        )
        depth_limited = False
        reported_samples_per_edge = samples_per_edge

    if not samples:
        empty_failure_reason = (
            "box_outside_camera_fov"
            if any(
                corner.z >= near_plane_m
                for corner in corners_camera
            )
            else "box_behind_near_plane"
        )

        return ActorCameraProjection(
            camera_name=calibration.camera_name,
            track_id=str(track_id),
            actor_class=str(actor_class),
            corner_count=len(corners_camera),
            edge_count=len(BOX_EDGE_INDEX_PAIRS),
            edge_samples_per_edge=reported_samples_per_edge,
            camera_sample_count=0,
            positive_depth_sample_count=0,
            within_fov_sample_count=0,
            inside_image_sample_count=0,
            projected_bbox=None,
            clipped_bbox=None,
            projected_area_px=0.0,
            inside_image_area_px=0.0,
            inside_image_ratio=0.0,
            projected_hull=(),
            clipped_hull=(),
            projected_hull_area_px=0.0,
            inside_image_hull_area_px=0.0,
            inside_image_hull_ratio=0.0,
            projected_height_px=0.0,
            minimum_depth_m=None,
            maximum_depth_m=None,
            truncated=False,
            projection_valid=False,
            failure_reason=empty_failure_reason,
        )

    positive_depth_count = sum(item.positive_z for item in projections)
    within_fov = [
        item for item in projections
        if item.positive_z and item.within_fov
        and math.isfinite(item.u) and math.isfinite(item.v)
    ]
    inside_image_count = sum(item.valid for item in projections)

    projected_points = [(item.u, item.v) for item in within_fov]
    projected_bbox = _bbox(projected_points)
    if projected_bbox is None:
        return ActorCameraProjection(
            camera_name=calibration.camera_name,
            track_id=str(track_id),
            actor_class=str(actor_class),
            corner_count=len(corners_camera),
            edge_count=len(BOX_EDGE_INDEX_PAIRS),
            edge_samples_per_edge=reported_samples_per_edge,
            camera_sample_count=len(samples),
            positive_depth_sample_count=positive_depth_count,
            within_fov_sample_count=0,
            inside_image_sample_count=inside_image_count,
            projected_bbox=None,
            clipped_bbox=None,
            projected_area_px=0.0,
            inside_image_area_px=0.0,
            inside_image_ratio=0.0,
            projected_hull=(),
            clipped_hull=(),
            projected_hull_area_px=0.0,
            inside_image_hull_area_px=0.0,
            inside_image_hull_ratio=0.0,
            projected_height_px=0.0,
            minimum_depth_m=min(point.z for point in samples),
            maximum_depth_m=max(point.z for point in samples),
            truncated=False,
            projection_valid=False,
            failure_reason="box_outside_camera_fov",
        )

    clipped_bbox = _clip_bbox_to_image(
        projected_bbox,
        width=calibration.width,
        height=calibration.height,
    )
    projected_area = projected_bbox.area
    inside_area = 0.0 if clipped_bbox is None else clipped_bbox.area
    inside_ratio = (
        inside_area / projected_area if projected_area > 0.0 else 0.0
    )
    hull = summarize_projected_hull(
        [Point2D(item.u, item.v) for item in within_fov],
        width=calibration.width,
        height=calibration.height,
    )
    truncated = hull.truncated_by_image
    valid = (
        clipped_bbox is not None
        and inside_area > 0.0
        and hull.inside_image_hull_area_px > 0.0
        and not depth_limited
    )

    return ActorCameraProjection(
        camera_name=calibration.camera_name,
        track_id=str(track_id),
        actor_class=str(actor_class),
        corner_count=len(corners_camera),
        edge_count=len(BOX_EDGE_INDEX_PAIRS),
        edge_samples_per_edge=reported_samples_per_edge,
        camera_sample_count=len(samples),
        positive_depth_sample_count=positive_depth_count,
        within_fov_sample_count=len(within_fov),
        inside_image_sample_count=inside_image_count,
        projected_bbox=projected_bbox,
        clipped_bbox=clipped_bbox,
        projected_area_px=projected_area,
        inside_image_area_px=inside_area,
        inside_image_ratio=inside_ratio,
        projected_hull=hull.projected_hull,
        clipped_hull=hull.clipped_hull,
        projected_hull_area_px=hull.projected_hull_area_px,
        inside_image_hull_area_px=hull.inside_image_hull_area_px,
        inside_image_hull_ratio=hull.inside_image_hull_ratio,
        projected_height_px=(0.0 if clipped_bbox is None else clipped_bbox.height),
        minimum_depth_m=min(point.z for point in samples),
        maximum_depth_m=max(point.z for point in samples),
        truncated=truncated,
        projection_valid=valid,
        failure_reason=(None if valid else ("adaptive_depth_limit_reached" if depth_limited else "projected_box_outside_image")),
    )


def project_actor_box_to_camera(
    actor: Mapping[str, Any],
    *,
    recorded_ego_message: Mapping[str, Any],
    calibration: FthetaCameraCalibration,
    samples_per_edge: int | None = None,
    near_plane_m: float = 1e-3,
    maximum_chord_error_px: float = 1.0,
    maximum_adaptive_depth: int = 14,
) -> ActorCameraProjection:
    """Transform one Actor box from map to rig to camera and project it."""
    corners_rig = actor_box_corners_in_rig(
        actor,
        recorded_ego_message=recorded_ego_message,
    )
    corners_camera = tuple(
        calibration.rig_point_to_camera(point) for point in corners_rig
    )
    return summarize_camera_box_projection(
        corners_camera,
        calibration=calibration,
        track_id=str(actor.get("track_id", "")),
        actor_class=str(actor.get("label_class", "")),
        samples_per_edge=samples_per_edge,
        near_plane_m=near_plane_m,
        maximum_chord_error_px=maximum_chord_error_px,
        maximum_adaptive_depth=maximum_adaptive_depth,
    )
