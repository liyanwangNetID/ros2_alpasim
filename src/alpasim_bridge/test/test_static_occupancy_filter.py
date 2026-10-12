from alpasim_bridge.static_occupancy_filter import (
    filter_drivable_columns,
    points_in_polygon,
)
from alpasim_msgs.msg import MapLane, MapPolyline, StaticOccupancy, VectorMap
from geometry_msgs.msg import Point
import numpy as np


def polyline(points):
    result = MapPolyline()
    result.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in points]
    return result


def test_points_in_polygon():
    polygon = np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    result = points_in_polygon(np.asarray([[1, 1], [3, 1]], dtype=float), polygon)
    assert result.tolist() == [True, False]


def test_filter_removes_only_road_connected_column():
    lane = MapLane()
    lane.left_boundary = polyline([(0, 1, 0), (4, 1, 0)])
    lane.right_boundary = polyline([(0, -1, 0), (4, -1, 0)])
    vector_map = VectorMap()
    vector_map.frame_id = "map"
    vector_map.scene_id = "scene"
    vector_map.lanes = [lane]
    raw = StaticOccupancy()
    raw.schema_version = "0.1"
    raw.scene_id = "scene"
    raw.scene_uuid = "uuid"
    raw.frame_id = "map"
    raw.representation = "single_interval_2p5d_static_occupancy"
    raw.grid_size_xy_m = 1.0
    raw.revision = 1
    raw.grid_i = [0, 1, 5]
    raw.grid_j = [0, 0, 0]
    raw.min_z = [0.1, 2.0, 0.0]
    raw.max_z = [4.0, 3.0, 2.0]
    raw.point_count = [3, 3, 3]
    raw.flags = [0, 0, 0]
    filtered, stats = filter_drivable_columns(raw, vector_map, 0.3, 0.0)
    assert list(filtered.grid_i) == [1, 5]
    assert stats["removed_road_connected_columns"] == 1



def test_road_height_is_interpolated_along_segment():
    from alpasim_bridge.static_occupancy_filter import nearest_reference_z

    query = np.asarray([[5.0, 0.0]], dtype=float)
    start = np.asarray([[0.0, 0.0, 0.0]], dtype=float)
    end = np.asarray([[10.0, 0.0, 2.0]], dtype=float)
    result = nearest_reference_z(query, start, end)
    np.testing.assert_allclose(result, [1.0])


def test_polygon_margin_includes_overlapping_grid_center():
    from alpasim_bridge.static_occupancy_filter import points_in_polygon_with_margin

    polygon = np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
    points = np.asarray([[2.15, 1.0], [2.25, 1.0]], dtype=float)
    result = points_in_polygon_with_margin(points, polygon, 0.20)
    assert result.tolist() == [True, False]



def test_two_meter_clearance_removes_roadside_reconstruction_but_keeps_bridge():
    lane = MapLane()
    lane.left_boundary = polyline([(0, 1, 0), (6, 1, 0)])
    lane.right_boundary = polyline([(0, -1, 0), (6, -1, 0)])
    vector_map = VectorMap()
    vector_map.frame_id = "map"
    vector_map.scene_id = "scene"
    vector_map.lanes = [lane]

    raw = StaticOccupancy()
    raw.schema_version = "0.1"
    raw.scene_id = "scene"
    raw.scene_uuid = "uuid"
    raw.frame_id = "map"
    raw.representation = "single_interval_2p5d_static_occupancy"
    raw.grid_size_xy_m = 1.0
    raw.revision = 1
    raw.grid_i = [0, 1, 2]
    raw.grid_j = [0, 0, 0]
    raw.min_z = [0.2, 1.4, 4.5]
    raw.max_z = [3.0, 4.0, 5.5]
    raw.point_count = [3, 3, 3]
    raw.flags = [0, 0, 0]

    filtered, stats = filter_drivable_columns(raw, vector_map, 2.0, 0.2)

    assert list(filtered.grid_i) == [2]
    assert stats["removed_road_connected_columns"] == 2
    assert stats["retained_suspended_drivable_columns"] == 1



def test_three_meter_clearance_preserves_high_bridge_only():
    lane = MapLane()
    lane.left_boundary = polyline([(0, 1, 0), (8, 1, 0)])
    lane.right_boundary = polyline([(0, -1, 0), (8, -1, 0)])
    vector_map = VectorMap()
    vector_map.frame_id = "map"
    vector_map.scene_id = "scene"
    vector_map.lanes = [lane]

    raw = StaticOccupancy()
    raw.schema_version = "0.1"
    raw.scene_id = "scene"
    raw.scene_uuid = "uuid"
    raw.frame_id = "map"
    raw.representation = "single_interval_2p5d_static_occupancy"
    raw.grid_size_xy_m = 1.0
    raw.revision = 1
    raw.grid_i = [0, 1, 2, 3]
    raw.grid_j = [0, 0, 0, 0]
    raw.min_z = [0.2, 1.8, 2.9, 4.5]
    raw.max_z = [2.0, 3.0, 4.0, 5.5]
    raw.point_count = [3, 3, 3, 3]
    raw.flags = [0, 0, 0, 0]

    filtered, stats = filter_drivable_columns(raw, vector_map, 3.0, 0.4)

    assert list(filtered.grid_i) == [3]
    assert stats["removed_road_connected_columns"] == 3
    assert stats["retained_suspended_drivable_columns"] == 1
