"""Step 7 camera projection domain.

Contains calibration parsing, rigid transforms, Actor box geometry, adaptive F-theta edge sampling, projected hulls, and image projection."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from typing import Any, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Sequence
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
        shifted = Vector3(point.x - self.rig_to_camera_translation.x, point.y - self.rig_to_camera_translation.y, point.z - self.rig_to_camera_translation.z)
        return rotate_vector_by_quaternion_inverse(shifted, self.rig_to_camera_rotation)

    def camera_point_to_rig(self, point: Vector3) -> Vector3:
        """Transform a camera optical-frame point into the rig/base frame."""
        rotated = rotate_vector_by_quaternion(point, self.rig_to_camera_rotation)
        return Vector3(rotated.x + self.rig_to_camera_translation.x, rotated.y + self.rig_to_camera_translation.y, rotated.z + self.rig_to_camera_translation.z)

    def project_camera_point(self, point: Vector3) -> PixelProjection:
        """Project one camera-frame point into the raw F-theta image."""
        return project_ftheta_point(point, width=self.width, height=self.height, principal_point_x=self.principal_point_x, principal_point_y=self.principal_point_y, angle_to_pixeldist_poly=self.angle_to_pixeldist_poly, max_angle_rad=self.max_angle_rad, linear_c=self.linear_c, linear_d=self.linear_d, linear_e=self.linear_e)

    def project_rig_point(self, point: Vector3) -> PixelProjection:
        """Transform a rig point to camera coordinates and project it."""
        return self.project_camera_point(self.rig_point_to_camera(point))

def _camera_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result

def _camera_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{name} must be a mapping')
    return value

def _camera_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()

def _camera_polynomial(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f'{name} must be a numeric sequence')
    result = tuple((_camera_finite(item, f'{name}[{index}]') for index, item in enumerate(value)))
    if not result:
        raise ValueError(f'{name} must not be empty')
    return result

def normalize_quaternion(quaternion: Quaternion) -> Quaternion:
    norm = math.sqrt(quaternion.x * quaternion.x + quaternion.y * quaternion.y + quaternion.z * quaternion.z + quaternion.w * quaternion.w)
    if norm <= _EPSILON:
        raise ValueError('quaternion norm must be positive')
    return Quaternion(quaternion.x / norm, quaternion.y / norm, quaternion.z / norm, quaternion.w / norm)

def quaternion_conjugate(quaternion: Quaternion) -> Quaternion:
    normalized = normalize_quaternion(quaternion)
    return Quaternion(-normalized.x, -normalized.y, -normalized.z, normalized.w)

def rotate_vector_by_quaternion(vector: Vector3, quaternion: Quaternion) -> Vector3:
    """Rotate a vector using a normalized x/y/z/w quaternion."""
    q = normalize_quaternion(quaternion)
    qv_x = q.y * vector.z - q.z * vector.y
    qv_y = q.z * vector.x - q.x * vector.z
    qv_z = q.x * vector.y - q.y * vector.x
    t_x = 2.0 * qv_x
    t_y = 2.0 * qv_y
    t_z = 2.0 * qv_z
    return Vector3(vector.x + q.w * t_x + (q.y * t_z - q.z * t_y), vector.y + q.w * t_y + (q.z * t_x - q.x * t_z), vector.z + q.w * t_z + (q.x * t_y - q.y * t_x))

def rotate_vector_by_quaternion_inverse(vector: Vector3, quaternion: Quaternion) -> Vector3:
    return rotate_vector_by_quaternion(vector, quaternion_conjugate(quaternion))

def evaluate_polynomial_low_to_high(value: float, coefficients: Sequence[float]) -> float:
    """Evaluate c0 + c1*x + ... using Horner's method."""
    x = _camera_finite(value, 'polynomial value')
    parsed = _camera_polynomial(coefficients, 'polynomial coefficients')
    result = 0.0
    for coefficient in reversed(parsed):
        result = result * x + coefficient
    return result

def project_ftheta_point(point: Vector3, *, width: int, height: int, principal_point_x: float, principal_point_y: float, angle_to_pixeldist_poly: Sequence[float], max_angle_rad: float | None, linear_c: float=1.0, linear_d: float=0.0, linear_e: float=0.0) -> PixelProjection:
    """Project one camera optical-frame point into an F-theta image."""
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError('width must be a positive integer')
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError('height must be a positive integer')
    x = _camera_finite(point.x, 'point.x')
    y = _camera_finite(point.y, 'point.y')
    z = _camera_finite(point.z, 'point.z')
    cx = _camera_finite(principal_point_x, 'principal_point_x')
    cy = _camera_finite(principal_point_y, 'principal_point_y')
    c = _camera_finite(linear_c, 'linear_c')
    d = _camera_finite(linear_d, 'linear_d')
    e = _camera_finite(linear_e, 'linear_e')
    coefficients = _camera_polynomial(angle_to_pixeldist_poly, 'angle_to_pixeldist_poly')
    positive_z = z > 0.0
    if not positive_z:
        return PixelProjection(u=math.nan, v=math.nan, valid=False, positive_z=False, within_fov=False, in_image=False, theta_rad=None, pixel_radius=None, failure_reason='behind_or_on_camera_plane')
    xy_norm = math.hypot(x, y)
    theta = math.atan2(xy_norm, z)
    if max_angle_rad is None:
        within_fov = True
    else:
        maximum = _camera_finite(max_angle_rad, 'max_angle_rad')
        within_fov = theta <= maximum + 1e-06
    radius = evaluate_polynomial_low_to_high(theta, coefficients)
    if xy_norm <= 1e-09:
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
    in_image = u >= -0.5 and u <= width - 0.5 and (v >= -0.5) and (v <= height - 0.5)
    if not within_fov:
        reason = 'outside_max_angle'
    elif not in_image:
        reason = 'outside_image'
    else:
        reason = None
    return PixelProjection(u=u, v=v, valid=positive_z and within_fov and in_image, positive_z=positive_z, within_fov=within_fov, in_image=in_image, theta_rad=theta, pixel_radius=radius, failure_reason=reason)

def parse_camera_calibration(calibration: Mapping[str, Any], *, camera_name: str, source_width: int | None=None, source_height: int | None=None) -> FthetaCameraCalibration:
    """Parse and optionally scale one recorded camera calibration.

    Scaling duplicates the installed AlpaSim Driver behavior. It is applied only
    when the actual source image resolution differs from the native calibration
    resolution.
    """
    name = _camera_nonempty_string(camera_name, 'camera_name')
    available = _camera_mapping(calibration.get('available_camera'), 'available_camera')
    intrinsics = _camera_mapping(available.get('intrinsics'), 'intrinsics')
    ftheta = _camera_mapping(intrinsics.get('ftheta_param'), 'ftheta_param')
    pose = _camera_mapping(available.get('rig_to_camera'), 'rig_to_camera')
    translation = _camera_mapping(pose.get('vec'), 'rig_to_camera.vec')
    rotation = _camera_mapping(pose.get('quat'), 'rig_to_camera.quat')
    linear = _camera_mapping(ftheta.get('linear_cde', {}), 'linear_cde')
    native_width = int(intrinsics['resolution_w'])
    native_height = int(intrinsics['resolution_h'])
    if native_width <= 0 or native_height <= 0:
        raise ValueError('native camera resolution must be positive')
    width = native_width if source_width is None else int(source_width)
    height = native_height if source_height is None else int(source_height)
    if width <= 0 or height <= 0:
        raise ValueError('source camera resolution must be positive')
    angle_poly = _camera_polynomial(ftheta.get('angle_to_pixeldist_poly'), 'angle_to_pixeldist_poly')
    inverse_poly = _camera_polynomial(ftheta.get('pixeldist_to_angle_poly'), 'pixeldist_to_angle_poly')
    principal_x = _camera_finite(ftheta.get('principal_point_x'), 'principal_point_x')
    principal_y = _camera_finite(ftheta.get('principal_point_y'), 'principal_point_y')
    linear_c = _camera_finite(linear.get('linear_c', 1.0), 'linear_c')
    linear_d = _camera_finite(linear.get('linear_d', 0.0), 'linear_d')
    linear_e = _camera_finite(linear.get('linear_e', 0.0), 'linear_e')
    if width != native_width or height != native_height:
        scale_x = width / native_width
        scale_y = height / native_height
        principal_x *= scale_x
        principal_y *= scale_y
        angle_poly = tuple((coefficient * scale_y for coefficient in angle_poly))
        inverse_poly = tuple((coefficient / scale_y ** power for power, coefficient in enumerate(inverse_poly)))
        x_ratio = scale_x / scale_y
        linear_c *= x_ratio
        linear_d *= x_ratio
    max_angle_value = _camera_finite(ftheta.get('max_angle', 0.0), 'max_angle')
    max_angle = max_angle_value if max_angle_value > 0.0 else None
    quaternion = normalize_quaternion(Quaternion(_camera_finite(rotation.get('x'), 'rig_to_camera.quat.x'), _camera_finite(rotation.get('y'), 'rig_to_camera.quat.y'), _camera_finite(rotation.get('z'), 'rig_to_camera.quat.z'), _camera_finite(rotation.get('w'), 'rig_to_camera.quat.w')))
    return FthetaCameraCalibration(camera_name=name, logical_id=_camera_nonempty_string(available.get('logical_id'), 'available_camera.logical_id'), width=width, height=height, principal_point_x=principal_x, principal_point_y=principal_y, angle_to_pixeldist_poly=angle_poly, pixeldist_to_angle_poly=inverse_poly, reference_poly=str(ftheta.get('reference_poly', '')), max_angle_rad=max_angle, linear_c=linear_c, linear_d=linear_d, linear_e=linear_e, rig_to_camera_translation=Vector3(_camera_finite(translation.get('x'), 'rig_to_camera.vec.x'), _camera_finite(translation.get('y'), 'rig_to_camera.vec.y'), _camera_finite(translation.get('z'), 'rig_to_camera.vec.z')), rig_to_camera_rotation=quaternion)

def load_camera_calibration(path: str | Path, *, camera_name: str, source_width: int | None=None, source_height: int | None=None) -> FthetaCameraCalibration:
    """Load and parse one recorded per-Clip camera calibration JSON file."""
    calibration_path = Path(path)
    value = json.loads(calibration_path.read_text(encoding='utf-8'))
    if not isinstance(value, Mapping):
        raise TypeError('camera calibration JSON root must be an object')
    return parse_camera_calibration(value, camera_name=camera_name, source_width=source_width, source_height=source_height)

@dataclass(frozen=True, slots=True)
class Pose3D:
    position: Vector3
    orientation: Quaternion

@dataclass(frozen=True, slots=True)
class ActorBox3D:
    track_id: str
    actor_class: str
    pose_map: Pose3D
    dimensions: Vector3
    corners_map: tuple[Vector3, ...]

def _box_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{name} must be numeric')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result

def _box_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f'{name} must be a mapping')
    return value

def _box_required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a non-empty string')
    return value.strip()

def parse_pose3d(pose: Mapping[str, Any], *, name: str) -> Pose3D:
    position = _box_mapping(pose.get('position'), f'{name}.position')
    orientation = _box_mapping(pose.get('orientation'), f'{name}.orientation')
    return Pose3D(position=Vector3(_box_finite(position.get('x'), f'{name}.position.x'), _box_finite(position.get('y'), f'{name}.position.y'), _box_finite(position.get('z'), f'{name}.position.z')), orientation=normalize_quaternion(Quaternion(_box_finite(orientation.get('x'), f'{name}.orientation.x'), _box_finite(orientation.get('y'), f'{name}.orientation.y'), _box_finite(orientation.get('z'), f'{name}.orientation.z'), _box_finite(orientation.get('w'), f'{name}.orientation.w'))))

def parse_recorded_ego_pose3d(message: Mapping[str, Any]) -> Pose3D:
    if message.get('pose_frame_id') != 'map':
        raise ValueError("recorded Ego pose_frame_id must be 'map'")
    return parse_pose3d({'position': _box_mapping(message.get('position'), 'recorded_ego.position'), 'orientation': _box_mapping(message.get('orientation'), 'recorded_ego.orientation')}, name='recorded_ego.pose')

def local_box_corners(dimensions: Vector3) -> tuple[Vector3, ...]:
    """Return eight centered corners for full x/y/z Actor dimensions."""
    length = _box_finite(dimensions.x, 'dimensions.x')
    width = _box_finite(dimensions.y, 'dimensions.y')
    height = _box_finite(dimensions.z, 'dimensions.z')
    if length <= 0.0 or width <= 0.0 or height <= 0.0:
        raise ValueError('Actor dimensions must all be positive')
    half_x = 0.5 * length
    half_y = 0.5 * width
    half_z = 0.5 * height
    return tuple((Vector3(x_sign * half_x, y_sign * half_y, z_sign * half_z) for z_sign in (-1.0, 1.0) for y_sign in (-1.0, 1.0) for x_sign in (-1.0, 1.0)))

def transform_local_point(pose: Pose3D, point: Vector3) -> Vector3:
    rotated = rotate_vector_by_quaternion(point, pose.orientation)
    return Vector3(rotated.x + pose.position.x, rotated.y + pose.position.y, rotated.z + pose.position.z)

def map_point_to_rig(point_map: Vector3, ego_pose_map: Pose3D) -> Vector3:
    shifted = Vector3(point_map.x - ego_pose_map.position.x, point_map.y - ego_pose_map.position.y, point_map.z - ego_pose_map.position.z)
    return rotate_vector_by_quaternion_inverse(shifted, ego_pose_map.orientation)

def rig_point_to_map(point_rig: Vector3, ego_pose_map: Pose3D) -> Vector3:
    rotated = rotate_vector_by_quaternion(point_rig, ego_pose_map.orientation)
    return Vector3(rotated.x + ego_pose_map.position.x, rotated.y + ego_pose_map.position.y, rotated.z + ego_pose_map.position.z)

def build_actor_box3d(actor: Mapping[str, Any]) -> ActorBox3D:
    """Build one oriented Actor box in the map frame."""
    pose = parse_pose3d(_box_mapping(actor.get('pose'), 'actor.pose'), name='actor.pose')
    dimensions_mapping = _box_mapping(actor.get('dimensions'), 'actor.dimensions')
    dimensions = Vector3(_box_finite(dimensions_mapping.get('x'), 'actor.dimensions.x'), _box_finite(dimensions_mapping.get('y'), 'actor.dimensions.y'), _box_finite(dimensions_mapping.get('z'), 'actor.dimensions.z'))
    corners = tuple((transform_local_point(pose, corner) for corner in local_box_corners(dimensions)))
    return ActorBox3D(track_id=_box_required_string(actor.get('track_id'), 'actor.track_id'), actor_class=_box_required_string(actor.get('label_class'), 'actor.label_class'), pose_map=pose, dimensions=dimensions, corners_map=corners)

def actor_box_corners_in_rig(actor: Mapping[str, Any], *, recorded_ego_message: Mapping[str, Any]) -> tuple[Vector3, ...]:
    """Return the Actor box corners in the Anchor Ego rig/base frame."""
    box = build_actor_box3d(actor)
    ego_pose = parse_recorded_ego_pose3d(recorded_ego_message)
    return tuple((map_point_to_rig(corner, ego_pose) for corner in box.corners_map))

@dataclass(frozen=True, slots=True)
class AdaptiveProjectedEdge:
    camera_points: tuple[Vector3, ...]
    projections: tuple[PixelProjection, ...]
    maximum_observed_chord_error_px: float
    maximum_depth_reached: int
    stopped_by_depth_limit: bool

def _adaptive_interpolate(first: Vector3, second: Vector3, ratio: float) -> Vector3:
    return Vector3(first.x + ratio * (second.x - first.x), first.y + ratio * (second.y - first.y), first.z + ratio * (second.z - first.z))

def point_to_segment_distance_px(point_u: float, point_v: float, first_u: float, first_v: float, second_u: float, second_v: float) -> float:
    """Return Euclidean distance from a 2D point to a closed segment."""
    values = (point_u, point_v, first_u, first_v, second_u, second_v)
    if not all((math.isfinite(float(value)) for value in values)):
        raise ValueError('pixel coordinates must be finite')
    delta_u = second_u - first_u
    delta_v = second_v - first_v
    squared_length = delta_u * delta_u + delta_v * delta_v
    if squared_length <= 1e-18:
        return math.hypot(point_u - first_u, point_v - first_v)
    ratio = ((point_u - first_u) * delta_u + (point_v - first_v) * delta_v) / squared_length
    ratio = min(1.0, max(0.0, ratio))
    nearest_u = first_u + ratio * delta_u
    nearest_v = first_v + ratio * delta_v
    return math.hypot(point_u - nearest_u, point_v - nearest_v)

def _adaptive_projectable(projection: PixelProjection) -> bool:
    return projection.positive_z and projection.within_fov and math.isfinite(projection.u) and math.isfinite(projection.v)

def sample_projected_edge_adaptive(first_camera: Vector3, second_camera: Vector3, calibration: FthetaCameraCalibration, *, maximum_chord_error_px: float, maximum_depth: int) -> AdaptiveProjectedEdge:
    """Adaptively sample one positive-depth, in-FOV camera-frame edge.

    Both endpoints and every recursively examined midpoint must be projectable.
    Near-plane clipping remains the responsibility of the caller.
    """
    error_limit = float(maximum_chord_error_px)
    if not math.isfinite(error_limit) or error_limit <= 0.0:
        raise ValueError('maximum_chord_error_px must be positive and finite')
    if isinstance(maximum_depth, bool) or not isinstance(maximum_depth, int):
        raise TypeError('maximum_depth must be an integer')
    if maximum_depth < 0:
        raise ValueError('maximum_depth must be non-negative')
    first_projection = calibration.project_camera_point(first_camera)
    second_projection = calibration.project_camera_point(second_camera)
    if not _adaptive_projectable(first_projection) or not _adaptive_projectable(second_projection):
        raise ValueError('edge endpoints must be positive-depth and within FOV')
    maximum_observed_error = 0.0
    maximum_depth_reached = 0
    stopped_by_depth_limit = False

    def recurse(first_point: Vector3, first_pixel: PixelProjection, second_point: Vector3, second_pixel: PixelProjection, depth: int) -> list[tuple[Vector3, PixelProjection]]:
        nonlocal maximum_observed_error
        nonlocal maximum_depth_reached
        nonlocal stopped_by_depth_limit
        maximum_depth_reached = max(maximum_depth_reached, depth)
        midpoint = _adaptive_interpolate(first_point, second_point, 0.5)
        midpoint_pixel = calibration.project_camera_point(midpoint)
        if not _adaptive_projectable(midpoint_pixel):
            raise ValueError('edge midpoint is not positive-depth and within FOV')
        chord_error = point_to_segment_distance_px(midpoint_pixel.u, midpoint_pixel.v, first_pixel.u, first_pixel.v, second_pixel.u, second_pixel.v)
        maximum_observed_error = max(maximum_observed_error, chord_error)
        if chord_error <= error_limit:
            return [(first_point, first_pixel), (midpoint, midpoint_pixel), (second_point, second_pixel)]
        if depth >= maximum_depth:
            stopped_by_depth_limit = True
            return [(first_point, first_pixel), (midpoint, midpoint_pixel), (second_point, second_pixel)]
        left = recurse(first_point, first_pixel, midpoint, midpoint_pixel, depth + 1)
        right = recurse(midpoint, midpoint_pixel, second_point, second_pixel, depth + 1)
        return left[:-1] + right
    pairs = recurse(first_camera, first_projection, second_camera, second_projection, 0)
    return AdaptiveProjectedEdge(camera_points=tuple((pair[0] for pair in pairs)), projections=tuple((pair[1] for pair in pairs)), maximum_observed_chord_error_px=maximum_observed_error, maximum_depth_reached=maximum_depth_reached, stopped_by_depth_limit=stopped_by_depth_limit)

@dataclass(frozen=True, slots=True)
class Point2D:
    u: float
    v: float

@dataclass(frozen=True, slots=True)
class ProjectedHullGeometry:
    projected_hull: tuple[Point2D, ...]
    clipped_hull: tuple[Point2D, ...]
    projected_hull_area_px: float
    inside_image_hull_area_px: float
    inside_image_hull_ratio: float
    truncated_by_image: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

def _hull_finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be finite')
    return converted

def _hull_cross(origin: Point2D, first: Point2D, second: Point2D) -> float:
    return (first.u - origin.u) * (second.v - origin.v) - (first.v - origin.v) * (second.u - origin.u)

def convex_hull(points: Sequence[Point2D]) -> tuple[Point2D, ...]:
    """Return a counter-clockwise convex hull using the monotonic chain."""
    unique = sorted({Point2D(_hull_finite(point.u, 'point.u'), _hull_finite(point.v, 'point.v')) for point in points}, key=lambda point: (point.u, point.v))
    if len(unique) <= 1:
        return tuple(unique)
    lower: list[Point2D] = []
    for point in unique:
        while len(lower) >= 2 and _hull_cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[Point2D] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _hull_cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])

def polygon_area(points: Sequence[Point2D]) -> float:
    """Return the unsigned polygon area using the shoelace formula."""
    if len(points) < 3:
        return 0.0
    doubled = 0.0
    for first, second in zip(points, (*points[1:], points[0])):
        doubled += first.u * second.v - second.u * first.v
    return 0.5 * abs(doubled)

def _hull_clip_polygon(polygon: Sequence[Point2D], *, inside, intersection) -> tuple[Point2D, ...]:
    if not polygon:
        return ()
    output: list[Point2D] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside:
            if not previous_inside:
                output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside:
            output.append(intersection(previous, current))
        previous = current
        previous_inside = current_inside
    return tuple(output)

def clip_polygon_to_image(polygon: Sequence[Point2D], *, width: int, height: int) -> tuple[Point2D, ...]:
    """Clip a polygon to [0,width-1] x [0,height-1]."""
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError('width must be a positive integer')
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError('height must be a positive integer')
    right = float(width - 1)
    bottom = float(height - 1)
    result = tuple(polygon)
    result = _hull_clip_polygon(result, inside=lambda point: point.u >= 0.0, intersection=lambda first, second: Point2D(0.0, first.v + (second.v - first.v) * (0.0 - first.u) / (second.u - first.u)))
    result = _hull_clip_polygon(result, inside=lambda point: point.u <= right, intersection=lambda first, second: Point2D(right, first.v + (second.v - first.v) * (right - first.u) / (second.u - first.u)))
    result = _hull_clip_polygon(result, inside=lambda point: point.v >= 0.0, intersection=lambda first, second: Point2D(first.u + (second.u - first.u) * (0.0 - first.v) / (second.v - first.v), 0.0))
    result = _hull_clip_polygon(result, inside=lambda point: point.v <= bottom, intersection=lambda first, second: Point2D(first.u + (second.u - first.u) * (bottom - first.v) / (second.v - first.v), bottom))
    return result

def summarize_projected_hull(projected_points: Sequence[Point2D], *, width: int, height: int) -> ProjectedHullGeometry:
    """Build, clip, and measure a projected AABB hull."""
    hull = convex_hull(projected_points)
    clipped = clip_polygon_to_image(hull, width=width, height=height)
    projected_area = polygon_area(hull)
    inside_area = polygon_area(clipped)
    ratio = inside_area / projected_area if projected_area > 0.0 else 0.0
    return ProjectedHullGeometry(projected_hull=hull, clipped_hull=clipped, projected_hull_area_px=projected_area, inside_image_hull_area_px=inside_area, inside_image_hull_ratio=ratio, truncated_by_image=projected_area > 0.0 and inside_area < projected_area - 1e-09)
_EPSILON = 1e-12

@dataclass(frozen=True, slots=True)
class SegmentInterval:
    start_ratio: float
    end_ratio: float

def interpolate_vector3(first: Vector3, second: Vector3, ratio: float) -> Vector3:
    return Vector3(first.x + ratio * (second.x - first.x), first.y + ratio * (second.y - first.y), first.z + ratio * (second.z - first.z))

def _fovclip_quadratic_value(a: float, b: float, c: float, value: float) -> float:
    return (a * value + b) * value + c

def quadratic_nonpositive_interval(a: float, b: float, c: float) -> SegmentInterval | None:
    """Solve a*t^2+b*t+c <= 0 over t in [0,1]."""
    values = (a, b, c)
    if not all((math.isfinite(value) for value in values)):
        raise ValueError('quadratic coefficients must be finite')
    if abs(a) <= _EPSILON:
        if abs(b) <= _EPSILON:
            return SegmentInterval(0.0, 1.0) if c <= 0.0 else None
        boundary = -c / b
        if b > 0.0:
            start, end = (0.0, min(1.0, boundary))
        else:
            start, end = (max(0.0, boundary), 1.0)
        return SegmentInterval(start, end) if end >= start else None
    discriminant = b * b - 4.0 * a * c
    if discriminant < -_EPSILON:
        return SegmentInterval(0.0, 1.0) if _fovclip_quadratic_value(a, b, c, 0.5) <= 0.0 else None
    root_term = math.sqrt(max(0.0, discriminant))
    first_root = (-b - root_term) / (2.0 * a)
    second_root = (-b + root_term) / (2.0 * a)
    low_root = min(first_root, second_root)
    high_root = max(first_root, second_root)
    candidate_boundaries = [0.0, 1.0]
    if 0.0 < low_root < 1.0:
        candidate_boundaries.append(low_root)
    if 0.0 < high_root < 1.0:
        candidate_boundaries.append(high_root)
    candidate_boundaries = sorted(set(candidate_boundaries))
    valid_intervals: list[tuple[float, float]] = []
    for start, end in zip(candidate_boundaries, candidate_boundaries[1:]):
        midpoint = 0.5 * (start + end)
        if _fovclip_quadratic_value(a, b, c, midpoint) <= _EPSILON:
            valid_intervals.append((start, end))
    for boundary in candidate_boundaries:
        if abs(_fovclip_quadratic_value(a, b, c, boundary)) <= _EPSILON:
            valid_intervals.append((boundary, boundary))
    if not valid_intervals:
        return None
    return SegmentInterval(min((item[0] for item in valid_intervals)), max((item[1] for item in valid_intervals)))

def clip_segment_to_angular_fov(first: Vector3, second: Vector3, *, max_angle_rad: float | None) -> tuple[Vector3, Vector3] | None:
    """Clip a positive-depth segment to theta <= max_angle_rad.

    When max_angle_rad is None, the original segment is returned. Angles must
    lie strictly between zero and pi/2 for the convex-cone formulation.
    """
    if max_angle_rad is None:
        return (first, second)
    maximum = float(max_angle_rad)
    if not math.isfinite(maximum) or not 0.0 < maximum < math.pi / 2.0:
        raise ValueError('max_angle_rad must be in (0, pi/2)')
    dx = second.x - first.x
    dy = second.y - first.y
    dz = second.z - first.z
    tangent_squared = math.tan(maximum) ** 2
    a = dx * dx + dy * dy - tangent_squared * dz * dz
    b = 2.0 * (first.x * dx + first.y * dy - tangent_squared * first.z * dz)
    c = first.x * first.x + first.y * first.y - tangent_squared * first.z * first.z
    interval = quadratic_nonpositive_interval(a, b, c)
    if interval is None:
        return None
    return (interpolate_vector3(first, second, interval.start_ratio), interpolate_vector3(first, second, interval.end_ratio))
BOX_EDGE_INDEX_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7))

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

def _image_finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f'{name} must be finite')
    return converted

def _image_interpolate(first: Vector3, second: Vector3, ratio: float) -> Vector3:
    return Vector3(first.x + ratio * (second.x - first.x), first.y + ratio * (second.y - first.y), first.z + ratio * (second.z - first.z))

def clip_segment_to_positive_z(first: Vector3, second: Vector3, *, near_plane_m: float) -> tuple[Vector3, Vector3] | None:
    """Clip a camera-frame segment to z >= near_plane_m."""
    near = _image_finite(near_plane_m, 'near_plane_m')
    if near <= 0.0:
        raise ValueError('near_plane_m must be positive')
    first_inside = first.z >= near
    second_inside = second.z >= near
    if first_inside and second_inside:
        return (first, second)
    if not first_inside and (not second_inside):
        return None
    denominator = second.z - first.z
    if abs(denominator) <= 1e-12:
        return None
    ratio = (near - first.z) / denominator
    intersection = _image_interpolate(first, second, ratio)
    intersection = Vector3(intersection.x, intersection.y, near)
    if first_inside:
        return (first, intersection)
    return (intersection, second)

def sample_box_edges_camera(corners_camera: Sequence[Vector3], *, samples_per_edge: int=9, near_plane_m: float=0.001) -> tuple[Vector3, ...]:
    """Clip and sample all twelve box edges in the camera frame."""
    if len(corners_camera) != 8:
        raise ValueError('an Actor box must contain exactly eight corners')
    if isinstance(samples_per_edge, bool) or not isinstance(samples_per_edge, int):
        raise TypeError('samples_per_edge must be an integer')
    if samples_per_edge < 2:
        raise ValueError('samples_per_edge must be at least two')
    samples: list[Vector3] = []
    for first_index, second_index in BOX_EDGE_INDEX_PAIRS:
        clipped = clip_segment_to_positive_z(corners_camera[first_index], corners_camera[second_index], near_plane_m=near_plane_m)
        if clipped is None:
            continue
        first, second = clipped
        for index in range(samples_per_edge):
            ratio = index / (samples_per_edge - 1)
            samples.append(_image_interpolate(first, second, ratio))
    return tuple(samples)

def sample_box_edges_projected_adaptive(corners_camera: Sequence[Vector3], *, calibration: FthetaCameraCalibration, near_plane_m: float=0.001, maximum_chord_error_px: float=1.0, maximum_adaptive_depth: int=14) -> tuple[tuple[Vector3, ...], tuple[PixelProjection, ...], bool]:
    camera_points: list[Vector3] = []
    projections: list[PixelProjection] = []
    depth_limited = False
    for first_index, second_index in BOX_EDGE_INDEX_PAIRS:
        segment = clip_segment_to_positive_z(corners_camera[first_index], corners_camera[second_index], near_plane_m=near_plane_m)
        if segment is None:
            continue
        segment = clip_segment_to_angular_fov(segment[0], segment[1], max_angle_rad=calibration.max_angle_rad)
        if segment is None:
            continue
        sampled = sample_projected_edge_adaptive(segment[0], segment[1], calibration, maximum_chord_error_px=maximum_chord_error_px, maximum_depth=maximum_adaptive_depth)
        camera_points.extend(sampled.camera_points)
        projections.extend(sampled.projections)
        depth_limited |= sampled.stopped_by_depth_limit
    return (tuple(camera_points), tuple(projections), depth_limited)

def _image_bbox(points: Sequence[tuple[float, float]]) -> BoundingBox2D | None:
    if not points:
        return None
    return BoundingBox2D(min_u=min((point[0] for point in points)), min_v=min((point[1] for point in points)), max_u=max((point[0] for point in points)), max_v=max((point[1] for point in points)))

def _image_clip_bbox_to_image(bbox: BoundingBox2D, *, width: int, height: int) -> BoundingBox2D | None:
    clipped = BoundingBox2D(min_u=max(0.0, bbox.min_u), min_v=max(0.0, bbox.min_v), max_u=min(float(width - 1), bbox.max_u), max_v=min(float(height - 1), bbox.max_v))
    if clipped.max_u <= clipped.min_u or clipped.max_v <= clipped.min_v:
        return None
    return clipped

def summarize_camera_box_projection(corners_camera: Sequence[Vector3], *, calibration: FthetaCameraCalibration, track_id: str, actor_class: str, samples_per_edge: int | None=None, near_plane_m: float=0.001, maximum_chord_error_px: float=1.0, maximum_adaptive_depth: int=14) -> ActorCameraProjection:
    """Project a camera-frame 3D box and summarize its 2D geometry."""
    if samples_per_edge is None:
        samples, projections, depth_limited = sample_box_edges_projected_adaptive(corners_camera, calibration=calibration, near_plane_m=near_plane_m, maximum_chord_error_px=maximum_chord_error_px, maximum_adaptive_depth=maximum_adaptive_depth)
        reported_samples_per_edge = 0
    else:
        samples = sample_box_edges_camera(corners_camera, samples_per_edge=samples_per_edge, near_plane_m=near_plane_m)
        projections = tuple((calibration.project_camera_point(point) for point in samples))
        depth_limited = False
        reported_samples_per_edge = samples_per_edge
    if not samples:
        empty_failure_reason = 'box_outside_camera_fov' if any((corner.z >= near_plane_m for corner in corners_camera)) else 'box_behind_near_plane'
        return ActorCameraProjection(camera_name=calibration.camera_name, track_id=str(track_id), actor_class=str(actor_class), corner_count=len(corners_camera), edge_count=len(BOX_EDGE_INDEX_PAIRS), edge_samples_per_edge=reported_samples_per_edge, camera_sample_count=0, positive_depth_sample_count=0, within_fov_sample_count=0, inside_image_sample_count=0, projected_bbox=None, clipped_bbox=None, projected_area_px=0.0, inside_image_area_px=0.0, inside_image_ratio=0.0, projected_hull=(), clipped_hull=(), projected_hull_area_px=0.0, inside_image_hull_area_px=0.0, inside_image_hull_ratio=0.0, projected_height_px=0.0, minimum_depth_m=None, maximum_depth_m=None, truncated=False, projection_valid=False, failure_reason=empty_failure_reason)
    positive_depth_count = sum((item.positive_z for item in projections))
    within_fov = [item for item in projections if item.positive_z and item.within_fov and math.isfinite(item.u) and math.isfinite(item.v)]
    inside_image_count = sum((item.valid for item in projections))
    projected_points = [(item.u, item.v) for item in within_fov]
    projected_bbox = _image_bbox(projected_points)
    if projected_bbox is None:
        return ActorCameraProjection(camera_name=calibration.camera_name, track_id=str(track_id), actor_class=str(actor_class), corner_count=len(corners_camera), edge_count=len(BOX_EDGE_INDEX_PAIRS), edge_samples_per_edge=reported_samples_per_edge, camera_sample_count=len(samples), positive_depth_sample_count=positive_depth_count, within_fov_sample_count=0, inside_image_sample_count=inside_image_count, projected_bbox=None, clipped_bbox=None, projected_area_px=0.0, inside_image_area_px=0.0, inside_image_ratio=0.0, projected_hull=(), clipped_hull=(), projected_hull_area_px=0.0, inside_image_hull_area_px=0.0, inside_image_hull_ratio=0.0, projected_height_px=0.0, minimum_depth_m=min((point.z for point in samples)), maximum_depth_m=max((point.z for point in samples)), truncated=False, projection_valid=False, failure_reason='box_outside_camera_fov')
    clipped_bbox = _image_clip_bbox_to_image(projected_bbox, width=calibration.width, height=calibration.height)
    projected_area = projected_bbox.area
    inside_area = 0.0 if clipped_bbox is None else clipped_bbox.area
    inside_ratio = inside_area / projected_area if projected_area > 0.0 else 0.0
    hull = summarize_projected_hull([Point2D(item.u, item.v) for item in within_fov], width=calibration.width, height=calibration.height)
    truncated = hull.truncated_by_image
    valid = clipped_bbox is not None and inside_area > 0.0 and (hull.inside_image_hull_area_px > 0.0) and (not depth_limited)
    return ActorCameraProjection(camera_name=calibration.camera_name, track_id=str(track_id), actor_class=str(actor_class), corner_count=len(corners_camera), edge_count=len(BOX_EDGE_INDEX_PAIRS), edge_samples_per_edge=reported_samples_per_edge, camera_sample_count=len(samples), positive_depth_sample_count=positive_depth_count, within_fov_sample_count=len(within_fov), inside_image_sample_count=inside_image_count, projected_bbox=projected_bbox, clipped_bbox=clipped_bbox, projected_area_px=projected_area, inside_image_area_px=inside_area, inside_image_ratio=inside_ratio, projected_hull=hull.projected_hull, clipped_hull=hull.clipped_hull, projected_hull_area_px=hull.projected_hull_area_px, inside_image_hull_area_px=hull.inside_image_hull_area_px, inside_image_hull_ratio=hull.inside_image_hull_ratio, projected_height_px=0.0 if clipped_bbox is None else clipped_bbox.height, minimum_depth_m=min((point.z for point in samples)), maximum_depth_m=max((point.z for point in samples)), truncated=truncated, projection_valid=valid, failure_reason=None if valid else 'adaptive_depth_limit_reached' if depth_limited else 'projected_box_outside_image')

def project_actor_box_to_camera(actor: Mapping[str, Any], *, recorded_ego_message: Mapping[str, Any], calibration: FthetaCameraCalibration, samples_per_edge: int | None=None, near_plane_m: float=0.001, maximum_chord_error_px: float=1.0, maximum_adaptive_depth: int=14) -> ActorCameraProjection:
    """Transform one Actor box from map to rig to camera and project it."""
    corners_rig = actor_box_corners_in_rig(actor, recorded_ego_message=recorded_ego_message)
    corners_camera = tuple((calibration.rig_point_to_camera(point) for point in corners_rig))
    return summarize_camera_box_projection(corners_camera, calibration=calibration, track_id=str(actor.get('track_id', '')), actor_class=str(actor.get('label_class', '')), samples_per_edge=samples_per_edge, near_plane_m=near_plane_m, maximum_chord_error_px=maximum_chord_error_px, maximum_adaptive_depth=maximum_adaptive_depth)
