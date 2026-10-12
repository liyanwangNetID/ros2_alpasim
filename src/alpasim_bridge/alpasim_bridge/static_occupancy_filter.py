#!/usr/bin/env python3
"""Remove road-surface-connected columns from raw static occupancy."""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass

import numpy as np
import rclpy
from alpasim_msgs.msg import StaticOccupancy, VectorMap
from alpasim_msgs.srv import GetStaticOccupancy, GetVectorMap
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


@dataclass(frozen=True)
class LanePolygon:
    xy: np.ndarray
    reference_segment_start: np.ndarray
    reference_segment_end: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray


def polyline_xyz(polyline) -> np.ndarray:
    points = np.asarray(
        [[float(p.x), float(p.y), float(p.z)] for p in polyline.points],
        dtype=np.float64,
    )
    return points


def lane_polygon(lane) -> LanePolygon | None:
    left = polyline_xyz(lane.left_boundary)
    right = polyline_xyz(lane.right_boundary)
    if len(left) < 2 or len(right) < 2:
        return None
    vertices = np.concatenate([left, right[::-1]], axis=0)
    xy = vertices[:, :2]
    segment_start = np.concatenate([left[:-1], right[:-1]], axis=0)
    segment_end = np.concatenate([left[1:], right[1:]], axis=0)
    return LanePolygon(
        xy=xy,
        reference_segment_start=segment_start,
        reference_segment_end=segment_end,
        minimum=xy.min(axis=0),
        maximum=xy.max(axis=0),
    )


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Vectorized even-odd test; boundary-adjacent points are treated inside."""
    x = points[:, 0]
    y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    xj, yj = polygon[-1]
    for xi, yi in polygon:
        crosses = ((yi > y) != (yj > y)) & (
            x <= (xj - xi) * (y - yi) / ((yj - yi) + 1e-15) + xi
        )
        inside ^= crosses
        xj, yj = xi, yi
    return inside


def minimum_distance_to_polygon_edges(
    points_xy: np.ndarray,
    polygon_xy: np.ndarray,
) -> np.ndarray:
    """Return each point's minimum 2D distance to the polygon boundary."""
    minimum = np.full(len(points_xy), np.inf, dtype=np.float64)
    segment_start = polygon_xy
    segment_end = np.roll(polygon_xy, -1, axis=0)
    for start, end in zip(segment_start, segment_end, strict=True):
        direction = end - start
        denominator = float(np.dot(direction, direction))
        if denominator <= 1e-15:
            projected = np.repeat(start[None, :], len(points_xy), axis=0)
        else:
            parameter = np.clip(
                ((points_xy - start) @ direction) / denominator,
                0.0,
                1.0,
            )
            projected = start + parameter[:, None] * direction
        distance = np.linalg.norm(points_xy - projected, axis=1)
        minimum = np.minimum(minimum, distance)
    return minimum


def points_in_polygon_with_margin(
    points_xy: np.ndarray,
    polygon_xy: np.ndarray,
    margin_m: float,
) -> np.ndarray:
    inside = points_in_polygon(points_xy, polygon_xy)
    if margin_m <= 0.0 or np.all(inside):
        return inside
    outside_indices = np.flatnonzero(~inside)
    distances = minimum_distance_to_polygon_edges(
        points_xy[outside_indices],
        polygon_xy,
    )
    inside[outside_indices[distances <= margin_m]] = True
    return inside


def nearest_reference_z(
    points_xy: np.ndarray,
    segment_start_xyz: np.ndarray,
    segment_end_xyz: np.ndarray,
) -> np.ndarray:
    """Project each point to the nearest lane-boundary segment and interpolate Z."""
    output = np.empty(len(points_xy), dtype=np.float64)
    chunk_size = 4096
    start_xy = segment_start_xyz[:, :2]
    direction_xy = segment_end_xyz[:, :2] - start_xy
    length_squared = np.sum(direction_xy * direction_xy, axis=1)
    valid_length = np.maximum(length_squared, 1e-15)
    delta_z = segment_end_xyz[:, 2] - segment_start_xyz[:, 2]
    for offset in range(0, len(points_xy), chunk_size):
        chunk = points_xy[offset:offset + chunk_size]
        relative = chunk[:, None, :] - start_xy[None, :, :]
        parameter = np.clip(
            np.sum(relative * direction_xy[None, :, :], axis=2)
            / valid_length[None, :],
            0.0,
            1.0,
        )
        projected = (
            start_xy[None, :, :]
            + parameter[:, :, None] * direction_xy[None, :, :]
        )
        distance_squared = np.sum(
            (chunk[:, None, :] - projected) ** 2,
            axis=2,
        )
        nearest = np.argmin(distance_squared, axis=1)
        rows = np.arange(len(chunk))
        nearest_parameter = parameter[rows, nearest]
        output[offset:offset + len(chunk)] = (
            segment_start_xyz[nearest, 2]
            + nearest_parameter * delta_z[nearest]
        )
    return output


def filtered_checksum(message: StaticOccupancy) -> str:
    digest = hashlib.sha256()
    digest.update(message.scene_uuid.encode("utf-8"))
    digest.update(np.asarray(message.grid_i, dtype="<i4").tobytes())
    digest.update(np.asarray(message.grid_j, dtype="<i4").tobytes())
    digest.update(np.asarray(message.min_z, dtype="<f4").tobytes())
    digest.update(np.asarray(message.max_z, dtype="<f4").tobytes())
    digest.update(np.asarray(message.point_count, dtype="<u4").tobytes())
    digest.update(np.asarray(message.flags, dtype="<u2").tobytes())
    return digest.hexdigest()


def filter_drivable_columns(
    raw: StaticOccupancy,
    vector_map: VectorMap,
    road_base_tolerance_m: float,
    polygon_margin_m: float,
) -> tuple[StaticOccupancy, dict[str, int]]:
    if raw.frame_id != vector_map.frame_id:
        raise ValueError(
            f"Frame mismatch: occupancy={raw.frame_id!r}, map={vector_map.frame_id!r}"
        )
    count = len(raw.grid_i)
    arrays = {
        "grid_i": np.asarray(raw.grid_i, dtype=np.int32),
        "grid_j": np.asarray(raw.grid_j, dtype=np.int32),
        "min_z": np.asarray(raw.min_z, dtype=np.float32),
        "max_z": np.asarray(raw.max_z, dtype=np.float32),
        "point_count": np.asarray(raw.point_count, dtype=np.uint32),
        "flags": np.asarray(raw.flags, dtype=np.uint16),
    }
    if count == 0 or any(len(value) != count for value in arrays.values()):
        raise ValueError("Raw static occupancy arrays are empty or inconsistent")
    grid = float(raw.grid_size_xy_m)
    centers = np.column_stack(
        ((arrays["grid_i"] + 0.5) * grid, (arrays["grid_j"] + 0.5) * grid)
    )
    remove = np.zeros(count, dtype=bool)
    inside_any = np.zeros(count, dtype=bool)
    polygon_count = 0
    for lane in vector_map.lanes:
        polygon = lane_polygon(lane)
        if polygon is None:
            continue
        polygon_count += 1
        candidate = np.flatnonzero(
            (~remove)
            & (centers[:, 0] >= polygon.minimum[0] - polygon_margin_m)
            & (centers[:, 0] <= polygon.maximum[0] + polygon_margin_m)
            & (centers[:, 1] >= polygon.minimum[1] - polygon_margin_m)
            & (centers[:, 1] <= polygon.maximum[1] + polygon_margin_m)
        )
        if len(candidate) == 0:
            continue
        local_inside = points_in_polygon_with_margin(
            centers[candidate],
            polygon.xy,
            polygon_margin_m,
        )
        selected = candidate[local_inside]
        if len(selected) == 0:
            continue
        inside_any[selected] = True
        road_z = nearest_reference_z(
            centers[selected],
            polygon.reference_segment_start,
            polygon.reference_segment_end,
        )
        connected_to_road = (
            arrays["min_z"][selected].astype(np.float64)
            <= road_z + road_base_tolerance_m
        )
        remove[selected[connected_to_road]] = True
    keep = ~remove
    output = StaticOccupancy()
    output.schema_version = raw.schema_version
    output.scene_id = raw.scene_id
    output.scene_uuid = raw.scene_uuid
    output.frame_id = raw.frame_id
    output.representation = raw.representation + "_road_filtered"
    output.semantic_labels = raw.semantic_labels
    output.grid_size_xy_m = raw.grid_size_xy_m
    output.revision = raw.revision
    output.grid_i = arrays["grid_i"][keep].tolist()
    output.grid_j = arrays["grid_j"][keep].tolist()
    output.min_z = arrays["min_z"][keep].tolist()
    output.max_z = arrays["max_z"][keep].tolist()
    output.point_count = arrays["point_count"][keep].tolist()
    output.flags = arrays["flags"][keep].tolist()
    output.payload_sha256 = filtered_checksum(output)
    return output, {
        "raw_columns": count,
        "lane_polygons": polygon_count,
        "columns_inside_drivable": int(np.count_nonzero(inside_any)),
        "removed_road_connected_columns": int(np.count_nonzero(remove)),
        "retained_suspended_drivable_columns": int(
            np.count_nonzero(inside_any & ~remove)
        ),
        "filtered_columns": int(np.count_nonzero(keep)),
    }


class StaticOccupancyFilter(Node):
    def __init__(self) -> None:
        super().__init__("static_occupancy_filter")
        self.declare_parameter("raw_topic", "/alpasim/static_occupancy/raw")
        self.declare_parameter("filtered_topic", "/alpasim/static_occupancy")
        self.declare_parameter("map_service", "/alpasim/map/get_vector_map")
        self.declare_parameter("service_name", "/alpasim/static_occupancy/get")
        self.declare_parameter("road_base_tolerance_m", 3.00)
        self.declare_parameter("polygon_margin_m", 0.40)
        self.road_base_tolerance_m = float(
            self.get_parameter("road_base_tolerance_m").value
        )
        self.polygon_margin_m = float(self.get_parameter("polygon_margin_m").value)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(
            StaticOccupancy,
            str(self.get_parameter("filtered_topic").value),
            qos,
        )
        self.subscription = self.create_subscription(
            StaticOccupancy,
            str(self.get_parameter("raw_topic").value),
            self.raw_callback,
            qos,
        )
        self.map_client = self.create_client(
            GetVectorMap,
            str(self.get_parameter("map_service").value),
        )
        self.service = self.create_service(
            GetStaticOccupancy,
            str(self.get_parameter("service_name").value),
            self.handle_request,
        )
        self.lock = threading.RLock()
        self.raw_message: StaticOccupancy | None = None
        self.vector_map: VectorMap | None = None
        self.filtered_message: StaticOccupancy | None = None
        self.request_in_progress = False
        self.timer = self.create_timer(0.25, self.poll)
        self.get_logger().info(
            "Static occupancy road filter initialized: "
            f"road_base_tolerance={self.road_base_tolerance_m:.2f} m"
        )

    def raw_callback(self, message: StaticOccupancy) -> None:
        with self.lock:
            self.raw_message = message
            self.filtered_message = None
        self.try_filter()

    def poll(self) -> None:
        with self.lock:
            need_map = self.vector_map is None
        if need_map and not self.request_in_progress and self.map_client.service_is_ready():
            self.request_in_progress = True
            request = GetVectorMap.Request()
            request.requested_scene_id = ""
            request.known_revision = 0
            future = self.map_client.call_async(request)
            future.add_done_callback(self.map_response)
        self.try_filter()

    def map_response(self, future) -> None:
        self.request_in_progress = False
        try:
            response = future.result()
            if not response.success or response.not_modified:
                raise RuntimeError(response.message)
            with self.lock:
                self.vector_map = response.vector_map
                self.filtered_message = None
            self.try_filter()
        except Exception as exc:
            self.get_logger().warning(f"Could not obtain VectorMap for filtering: {exc}")

    def try_filter(self) -> None:
        with self.lock:
            raw = self.raw_message
            vector_map = self.vector_map
            existing = self.filtered_message
        if raw is None or vector_map is None:
            return
        if existing is not None and existing.scene_id == raw.scene_id:
            return
        if vector_map.scene_id and raw.scene_id != vector_map.scene_id:
            self.get_logger().warning(
                f"Waiting for matching VectorMap: occupancy={raw.scene_id}, "
                f"map={vector_map.scene_id}"
            )
            with self.lock:
                self.vector_map = None
            return
        try:
            filtered, stats = filter_drivable_columns(
                raw,
                vector_map,
                self.road_base_tolerance_m,
                self.polygon_margin_m,
            )
        except ValueError as exc:
            self.get_logger().error(f"Static occupancy filtering failed: {exc}")
            return
        with self.lock:
            filtered.revision = int(raw.revision)
            self.filtered_message = filtered
        self.publisher.publish(filtered)
        self.get_logger().info(
            "Published road-filtered static occupancy: "
            f"scene={filtered.scene_id}, raw={stats['raw_columns']}, "
            f"inside={stats['columns_inside_drivable']}, "
            f"removed={stats['removed_road_connected_columns']}, "
            f"suspended_inside={stats['retained_suspended_drivable_columns']}, "
            f"remaining={stats['filtered_columns']}, "
            f"lanes={stats['lane_polygons']}"
        )

    def handle_request(self, request, response):
        with self.lock:
            message = self.filtered_message
        if message is None:
            response.success = False
            response.message = "Road-filtered static occupancy is not ready"
            return response
        if request.requested_scene_id and request.requested_scene_id != message.scene_id:
            response.success = False
            response.message = "Requested scene does not match filtered occupancy"
            response.revision = message.revision
            return response
        response.success = True
        response.revision = message.revision
        if request.known_revision and request.known_revision == message.revision:
            response.not_modified = True
            response.message = "Client already knows the current filtered revision"
            return response
        response.not_modified = False
        response.message = "Returning road-filtered static occupancy"
        response.static_occupancy = message
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StaticOccupancyFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
