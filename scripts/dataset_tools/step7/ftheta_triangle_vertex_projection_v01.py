"""F-theta vertex projection records for camera-frame triangles.

This module projects only the three vertices of one near-clipped camera-frame
triangle. It does not rasterize triangle interiors, clip curved F-theta edges,
or decide Actor-to-Actor occlusion.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from step7.camera_projection_v01 import Vector3


class CameraPointProjector(Protocol):
    camera_name: str

    def project_camera_point(self, point: Vector3): ...


@dataclass(frozen=True, slots=True)
class ProjectedTriangleVertex:
    vertex_index: int
    camera_x_m: float
    camera_y_m: float
    camera_z_m: float
    u_px: float
    v_px: float
    positive_z: bool
    within_fov: bool
    inside_image: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FthetaTriangleVertexProjection:
    camera_name: str
    vertices: tuple[
        ProjectedTriangleVertex,
        ProjectedTriangleVertex,
        ProjectedTriangleVertex,
    ]
    all_vertices_within_fov: bool
    all_vertices_inside_image: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_name": self.camera_name,
            "vertices": [vertex.to_dict() for vertex in self.vertices],
            "all_vertices_within_fov": self.all_vertices_within_fov,
            "all_vertices_inside_image": self.all_vertices_inside_image,
        }


def project_ftheta_triangle_vertices(
    triangle_camera: Sequence[Vector3],
    *,
    calibration: CameraPointProjector,
    near_plane_m: float = 1e-3,
) -> FthetaTriangleVertexProjection:
    """Project the three vertices of one prepared camera-frame triangle.

    All input vertices must already satisfy z >= near_plane_m. Vertex-only
    flags are diagnostic and must not be interpreted as full-triangle clipping
    or rasterization results.
    """
    if len(triangle_camera) != 3:
        raise ValueError("a triangle must contain exactly three vertices")
    near = float(near_plane_m)
    if not math.isfinite(near) or near <= 0.0:
        raise ValueError("near_plane_m must be finite and positive")
    camera_name = getattr(calibration, "camera_name", None)
    if not isinstance(camera_name, str) or not camera_name:
        raise ValueError("calibration.camera_name must be a non-empty string")

    records: list[ProjectedTriangleVertex] = []
    for index, vertex in enumerate(triangle_camera):
        coordinates = (float(vertex.x), float(vertex.y), float(vertex.z))
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError("triangle vertices must be finite")
        if coordinates[2] < near:
            raise ValueError("triangle vertex is behind near_plane_m")

        projected = calibration.project_camera_point(vertex)
        u = float(projected.u)
        v = float(projected.v)
        if not math.isfinite(u) or not math.isfinite(v):
            raise ValueError("projected pixel coordinates must be finite")
        positive_z = bool(projected.positive_z)
        within_fov = bool(projected.within_fov)
        inside_image = bool(projected.valid)
        if not positive_z:
            raise ValueError("prepared triangle vertex projected with non-positive z")
        if inside_image and not within_fov:
            raise ValueError("inside-image projection must also be within FOV")

        records.append(
            ProjectedTriangleVertex(
                vertex_index=index,
                camera_x_m=coordinates[0],
                camera_y_m=coordinates[1],
                camera_z_m=coordinates[2],
                u_px=u,
                v_px=v,
                positive_z=positive_z,
                within_fov=within_fov,
                inside_image=inside_image,
            )
        )

    vertices = tuple(records)
    return FthetaTriangleVertexProjection(
        camera_name=camera_name,
        vertices=vertices,  # type: ignore[arg-type]
        all_vertices_within_fov=all(item.within_fov for item in vertices),
        all_vertices_inside_image=all(item.inside_image for item in vertices),
    )
