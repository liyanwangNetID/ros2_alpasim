#!/usr/bin/env python3
"""Step 7D camera calibration, rigid transforms, and F-theta ray projection.

Calibration is parsed per Clip and per camera. No camera extrinsic or intrinsic
parameter is hard-coded. The F-theta projection follows the installed
AlpaSim Driver convention:

- camera optical +z is forward;
- camera optical +x maps toward increasing image u;
- camera optical +y maps toward increasing image v;
- angle_to_pixeldist_poly coefficients are low-order to high-order;
- rig_to_camera stores the camera optical-frame pose in the rig/base frame.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class Quaternion:
    x: float
    y: float
    z: float
    w: float


@dataclass(frozen=True, slots=True)
class PixelProjection:
    u: float
    v: float
    valid: bool
    positive_z: bool
    within_fov: bool
    in_image: bool
    theta_rad: float | None
    pixel_radius: float | None
    failure_reason: str | None


@dataclass(frozen=True, slots=True)
class FthetaCameraCalibration:
    camera_name: str
    logical_id: str
    width: int
    height: int
    principal_point_x: float
    principal_point_y: float
    angle_to_pixeldist_poly: tuple[float, ...]
    pixeldist_to_angle_poly: tuple[float, ...]
    reference_poly: str
    max_angle_rad: float | None
    linear_c: float
    linear_d: float
    linear_e: float
    rig_to_camera_translation: Vector3
    rig_to_camera_rotation: Quaternion

    def rig_point_to_camera(self, point: Vector3) -> Vector3:
        """Transform a rig/base-frame point into the camera optical frame."""
        shifted = Vector3(
            point.x - self.rig_to_camera_translation.x,
            point.y - self.rig_to_camera_translation.y,
            point.z - self.rig_to_camera_translation.z,
        )
        return rotate_vector_by_quaternion_inverse(
            shifted,
            self.rig_to_camera_rotation,
        )

    def camera_point_to_rig(self, point: Vector3) -> Vector3:
        """Transform a camera optical-frame point into the rig/base frame."""
        rotated = rotate_vector_by_quaternion(
            point,
            self.rig_to_camera_rotation,
        )
        return Vector3(
            rotated.x + self.rig_to_camera_translation.x,
            rotated.y + self.rig_to_camera_translation.y,
            rotated.z + self.rig_to_camera_translation.z,
        )

    def project_camera_point(self, point: Vector3) -> PixelProjection:
        """Project one camera-frame point into the raw F-theta image."""
        return project_ftheta_point(
            point,
            width=self.width,
            height=self.height,
            principal_point_x=self.principal_point_x,
            principal_point_y=self.principal_point_y,
            angle_to_pixeldist_poly=self.angle_to_pixeldist_poly,
            max_angle_rad=self.max_angle_rad,
            linear_c=self.linear_c,
            linear_d=self.linear_d,
            linear_e=self.linear_e,
        )

    def project_rig_point(self, point: Vector3) -> PixelProjection:
        """Transform a rig point to camera coordinates and project it."""
        return self.project_camera_point(self.rig_point_to_camera(point))


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _polynomial(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    result = tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    norm = math.sqrt(
        quaternion.x * quaternion.x
        + quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
        + quaternion.w * quaternion.w
    )
    if norm <= _EPSILON:
        raise ValueError("quaternion norm must be positive")
    return Quaternion(
        quaternion.x / norm,
        quaternion.y / norm,
        quaternion.z / norm,
        quaternion.w / norm,
    )


def quaternion_conjugate(quaternion: Quaternion) -> Quaternion:
    normalized = normalize_quaternion(quaternion)
    return Quaternion(
        -normalized.x,
        -normalized.y,
        -normalized.z,
        normalized.w,
    )


def rotate_vector_by_quaternion(
    vector: Vector3,
    quaternion: Quaternion,
) -> Vector3:
    """Rotate a vector using a normalized x/y/z/w quaternion."""
    q = normalize_quaternion(quaternion)
    qv_x = q.y * vector.z - q.z * vector.y
    qv_y = q.z * vector.x - q.x * vector.z
    qv_z = q.x * vector.y - q.y * vector.x
    t_x = 2.0 * qv_x
    t_y = 2.0 * qv_y
    t_z = 2.0 * qv_z
    return Vector3(
        vector.x + q.w * t_x + (q.y * t_z - q.z * t_y),
        vector.y + q.w * t_y + (q.z * t_x - q.x * t_z),
        vector.z + q.w * t_z + (q.x * t_y - q.y * t_x),
    )


def rotate_vector_by_quaternion_inverse(
    vector: Vector3,
    quaternion: Quaternion,
) -> Vector3:
    return rotate_vector_by_quaternion(
        vector,
        quaternion_conjugate(quaternion),
    )


def evaluate_polynomial_low_to_high(
    value: float,
    coefficients: Sequence[float],
) -> float:
    """Evaluate c0 + c1*x + ... using Horner's method."""
    x = _finite(value, "polynomial value")
    parsed = _polynomial(coefficients, "polynomial coefficients")
    result = 0.0
    for coefficient in reversed(parsed):
        result = result * x + coefficient
    return result


def project_ftheta_point(
    point: Vector3,
    *,
    width: int,
    height: int,
    principal_point_x: float,
    principal_point_y: float,
    angle_to_pixeldist_poly: Sequence[float],
    max_angle_rad: float | None,
    linear_c: float = 1.0,
    linear_d: float = 0.0,
    linear_e: float = 0.0,
) -> PixelProjection:
    """Project one camera optical-frame point into an F-theta image."""
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")

    x = _finite(point.x, "point.x")
    y = _finite(point.y, "point.y")
    z = _finite(point.z, "point.z")
    cx = _finite(principal_point_x, "principal_point_x")
    cy = _finite(principal_point_y, "principal_point_y")
    c = _finite(linear_c, "linear_c")
    d = _finite(linear_d, "linear_d")
    e = _finite(linear_e, "linear_e")
    coefficients = _polynomial(
        angle_to_pixeldist_poly,
        "angle_to_pixeldist_poly",
    )

    positive_z = z > 0.0
    if not positive_z:
        return PixelProjection(
            u=math.nan,
            v=math.nan,
            valid=False,
            positive_z=False,
            within_fov=False,
            in_image=False,
            theta_rad=None,
            pixel_radius=None,
            failure_reason="behind_or_on_camera_plane",
        )

    xy_norm = math.hypot(x, y)
    theta = math.atan2(xy_norm, z)
    if max_angle_rad is None:
        within_fov = True
    else:
        maximum = _finite(max_angle_rad, "max_angle_rad")
        within_fov = theta <= maximum + 1e-6

    radius = evaluate_polynomial_low_to_high(theta, coefficients)
    if xy_norm <= 1e-9:
        offset_x = 0.0
        offset_y = 0.0
    else:
        scale = radius / xy_norm
        offset_x = x * scale
        offset_y = y * scale

    pixel_offset_x = c * offset_x + d * offset_y
    pixel_offset_y = e * offset_x + offset_y
    u = pixel_offset_x + cx
    v = pixel_offset_y + cy
    in_image = (
        u >= -0.5
        and u <= width - 0.5
        and v >= -0.5
        and v <= height - 0.5
    )

    if not within_fov:
        reason = "outside_max_angle"
    elif not in_image:
        reason = "outside_image"
    else:
        reason = None

    return PixelProjection(
        u=u,
        v=v,
        valid=positive_z and within_fov and in_image,
        positive_z=positive_z,
        within_fov=within_fov,
        in_image=in_image,
        theta_rad=theta,
        pixel_radius=radius,
        failure_reason=reason,
    )


def parse_camera_calibration(
    calibration: Mapping[str, Any],
    *,
    camera_name: str,
    source_width: int | None = None,
    source_height: int | None = None,
) -> FthetaCameraCalibration:
    """Parse and optionally scale one recorded camera calibration.

    Scaling duplicates the installed AlpaSim Driver behavior. It is applied only
    when the actual source image resolution differs from the native calibration
    resolution.
    """
    name = _nonempty_string(camera_name, "camera_name")
    available = _mapping(calibration.get("available_camera"), "available_camera")
    intrinsics = _mapping(available.get("intrinsics"), "intrinsics")
    ftheta = _mapping(intrinsics.get("ftheta_param"), "ftheta_param")
    pose = _mapping(available.get("rig_to_camera"), "rig_to_camera")
    translation = _mapping(pose.get("vec"), "rig_to_camera.vec")
    rotation = _mapping(pose.get("quat"), "rig_to_camera.quat")
    linear = _mapping(ftheta.get("linear_cde", {}), "linear_cde")

    native_width = int(intrinsics["resolution_w"])
    native_height = int(intrinsics["resolution_h"])
    if native_width <= 0 or native_height <= 0:
        raise ValueError("native camera resolution must be positive")

    width = native_width if source_width is None else int(source_width)
    height = native_height if source_height is None else int(source_height)
    if width <= 0 or height <= 0:
        raise ValueError("source camera resolution must be positive")

    angle_poly = _polynomial(
        ftheta.get("angle_to_pixeldist_poly"),
        "angle_to_pixeldist_poly",
    )
    inverse_poly = _polynomial(
        ftheta.get("pixeldist_to_angle_poly"),
        "pixeldist_to_angle_poly",
    )
    principal_x = _finite(ftheta.get("principal_point_x"), "principal_point_x")
    principal_y = _finite(ftheta.get("principal_point_y"), "principal_point_y")
    linear_c = _finite(linear.get("linear_c", 1.0), "linear_c")
    linear_d = _finite(linear.get("linear_d", 0.0), "linear_d")
    linear_e = _finite(linear.get("linear_e", 0.0), "linear_e")

    if width != native_width or height != native_height:
        scale_x = width / native_width
        scale_y = height / native_height
        principal_x *= scale_x
        principal_y *= scale_y
        angle_poly = tuple(coefficient * scale_y for coefficient in angle_poly)
        inverse_poly = tuple(
            coefficient / (scale_y ** power)
            for power, coefficient in enumerate(inverse_poly)
        )
        x_ratio = scale_x / scale_y
        linear_c *= x_ratio
        linear_d *= x_ratio

    max_angle_value = _finite(ftheta.get("max_angle", 0.0), "max_angle")
    max_angle = max_angle_value if max_angle_value > 0.0 else None

    quaternion = normalize_quaternion(
        Quaternion(
            _finite(rotation.get("x"), "rig_to_camera.quat.x"),
            _finite(rotation.get("y"), "rig_to_camera.quat.y"),
            _finite(rotation.get("z"), "rig_to_camera.quat.z"),
            _finite(rotation.get("w"), "rig_to_camera.quat.w"),
        )
    )

    return FthetaCameraCalibration(
        camera_name=name,
        logical_id=_nonempty_string(
            available.get("logical_id"),
            "available_camera.logical_id",
        ),
        width=width,
        height=height,
        principal_point_x=principal_x,
        principal_point_y=principal_y,
        angle_to_pixeldist_poly=angle_poly,
        pixeldist_to_angle_poly=inverse_poly,
        reference_poly=str(ftheta.get("reference_poly", "")),
        max_angle_rad=max_angle,
        linear_c=linear_c,
        linear_d=linear_d,
        linear_e=linear_e,
        rig_to_camera_translation=Vector3(
            _finite(translation.get("x"), "rig_to_camera.vec.x"),
            _finite(translation.get("y"), "rig_to_camera.vec.y"),
            _finite(translation.get("z"), "rig_to_camera.vec.z"),
        ),
        rig_to_camera_rotation=quaternion,
    )


def load_camera_calibration(
    path: str | Path,
    *,
    camera_name: str,
    source_width: int | None = None,
    source_height: int | None = None,
) -> FthetaCameraCalibration:
    """Load and parse one recorded per-Clip camera calibration JSON file."""
    calibration_path = Path(path)
    value = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("camera calibration JSON root must be an object")
    return parse_camera_calibration(
        value,
        camera_name=camera_name,
        source_width=source_width,
        source_height=source_height,
    )
