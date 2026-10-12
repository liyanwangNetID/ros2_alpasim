from alpasim_bridge.static_occupancy_marker_publisher import height_bin


def test_height_bins_are_monotonic():
    assert [height_bin(value) for value in (0.1, 0.4, 0.8, 1.5, 3, 7, 12, 20)] == list(range(8))


def test_shallow_columns_are_hidden_from_rviz():
    from alpasim_msgs.msg import StaticOccupancy
    from alpasim_bridge.static_occupancy_marker_publisher import build_markers

    message = StaticOccupancy()
    message.frame_id = "map"
    message.grid_size_xy_m = 0.4
    message.grid_i = [0, 1]
    message.grid_j = [0, 1]
    message.min_z = [0.0, 0.0]
    message.max_z = [0.20, 1.00]

    markers = build_markers(message, (0.82, 0.84, 0.86, 0.38), 0.30)
    rendered_points = sum(len(marker.points) for marker in markers.markers)
    assert rendered_points == 1
