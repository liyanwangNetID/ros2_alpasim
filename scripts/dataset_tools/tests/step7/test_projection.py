"""Tests for the consolidated Step 7 projection domain."""
from __future__ import annotations
import math
import sys
import tempfile
import unittest
from pathlib import Path
from step7.projection import FthetaCameraCalibration, Quaternion, Vector3, evaluate_polynomial_low_to_high, load_camera_calibration, normalize_quaternion, parse_camera_calibration, project_ftheta_point, rotate_vector_by_quaternion
from project_paths import ALPASIM_DATA_ROOT
from step7.projection import Pose3D, actor_box_corners_in_rig, build_actor_box3d, local_box_corners, map_point_to_rig, rig_point_to_map
from step7.projection import Quaternion, Vector3
from step7.projection import point_to_segment_distance_px, sample_projected_edge_adaptive
from step7.projection import FthetaCameraCalibration, Quaternion, Vector3
from step7.projection import Point2D, clip_polygon_to_image, convex_hull, polygon_area, summarize_projected_hull
from step7.projection import Vector3
from step7.projection import clip_segment_to_angular_fov, quadratic_nonpositive_interval
from step7.projection import BOX_EDGE_INDEX_PAIRS, clip_segment_to_positive_z, sample_box_edges_camera, summarize_camera_box_projection

def camera_calibration_dict(*, quaternion: tuple[float, float, float, float]=(0.0, 0.0, 0.0, 1.0), translation: tuple[float, float, float]=(1.0, 2.0, 3.0)) -> dict:
    qx, qy, qz, qw = quaternion
    tx, ty, tz = translation
    return {'available_camera': {'logical_id': 'camera_test', 'intrinsics': {'logical_id': 'camera_test', 'resolution_w': 200, 'resolution_h': 100, 'shutter_type': 'ROLLING_TOP_TO_BOTTOM', 'ftheta_param': {'angle_to_pixeldist_poly': [0.0, 50.0], 'pixeldist_to_angle_poly': [0.0, 0.02], 'principal_point_x': 100.0, 'principal_point_y': 50.0, 'max_angle': 1.0, 'reference_poly': 'PIXELDIST_TO_ANGLE', 'linear_cde': {'linear_c': 1.0, 'linear_d': 0.0, 'linear_e': 0.0}}}, 'rig_to_camera': {'vec': {'x': tx, 'y': ty, 'z': tz}, 'quat': {'x': qx, 'y': qy, 'z': qz, 'w': qw}}}}

class PolynomialTests(unittest.TestCase):

    def test_low_to_high_order(self) -> None:
        self.assertEqual(evaluate_polynomial_low_to_high(2.0, [1.0, 3.0, 4.0]), 23.0)

class QuaternionTests(unittest.TestCase):

    def test_quaternion_is_normalized(self) -> None:
        result = normalize_quaternion(Quaternion(0.0, 0.0, 0.0, 5.0))
        self.assertEqual(result, Quaternion(0.0, 0.0, 0.0, 1.0))

    def test_rotation_about_z(self) -> None:
        half = math.pi / 4.0
        result = rotate_vector_by_quaternion(Vector3(1.0, 0.0, 0.0), Quaternion(0.0, 0.0, math.sin(half), math.cos(half)))
        self.assertAlmostEqual(result.x, 0.0, places=7)
        self.assertAlmostEqual(result.y, 1.0, places=7)
        self.assertAlmostEqual(result.z, 0.0, places=7)

class RigidTransformTests(unittest.TestCase):

    def test_rig_camera_round_trip(self) -> None:
        half = 0.5 * math.radians(30.0)
        calibration = parse_camera_calibration(camera_calibration_dict(quaternion=(0.0, 0.0, math.sin(half), math.cos(half))), camera_name='test')
        original = Vector3(7.0, -2.0, 5.0)
        camera = calibration.rig_point_to_camera(original)
        recovered = calibration.camera_point_to_rig(camera)
        self.assertAlmostEqual(recovered.x, original.x, places=7)
        self.assertAlmostEqual(recovered.y, original.y, places=7)
        self.assertAlmostEqual(recovered.z, original.z, places=7)

    def test_identity_rotation_subtracts_camera_translation(self) -> None:
        calibration = parse_camera_calibration(camera_calibration_dict(), camera_name='test')
        result = calibration.rig_point_to_camera(Vector3(1.0, 2.0, 8.0))
        self.assertEqual(result, Vector3(0.0, 0.0, 5.0))

class ProjectionTests(unittest.TestCase):

    def test_optical_axis_projects_to_principal_point(self) -> None:
        result = project_ftheta_point(Vector3(0.0, 0.0, 10.0), width=200, height=100, principal_point_x=100.0, principal_point_y=50.0, angle_to_pixeldist_poly=[0.0, 50.0], max_angle_rad=1.0)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.u, 100.0)
        self.assertAlmostEqual(result.v, 50.0)

    def test_positive_x_moves_pixel_right(self) -> None:
        result = project_ftheta_point(Vector3(1.0, 0.0, 10.0), width=200, height=100, principal_point_x=100.0, principal_point_y=50.0, angle_to_pixeldist_poly=[0.0, 50.0], max_angle_rad=1.0)
        self.assertTrue(result.valid)
        self.assertGreater(result.u, 100.0)
        self.assertAlmostEqual(result.v, 50.0)

    def test_positive_y_moves_pixel_down(self) -> None:
        result = project_ftheta_point(Vector3(0.0, 1.0, 10.0), width=200, height=100, principal_point_x=100.0, principal_point_y=50.0, angle_to_pixeldist_poly=[0.0, 50.0], max_angle_rad=1.0)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.u, 100.0)
        self.assertGreater(result.v, 50.0)

    def test_behind_camera_is_invalid(self) -> None:
        result = project_ftheta_point(Vector3(0.0, 0.0, -1.0), width=200, height=100, principal_point_x=100.0, principal_point_y=50.0, angle_to_pixeldist_poly=[0.0, 50.0], max_angle_rad=1.0)
        self.assertFalse(result.valid)
        self.assertEqual(result.failure_reason, 'behind_or_on_camera_plane')

    def test_outside_max_angle_is_invalid(self) -> None:
        result = project_ftheta_point(Vector3(10.0, 0.0, 1.0), width=1000, height=1000, principal_point_x=500.0, principal_point_y=500.0, angle_to_pixeldist_poly=[0.0, 50.0], max_angle_rad=0.5)
        self.assertFalse(result.valid)
        self.assertFalse(result.within_fov)
        self.assertEqual(result.failure_reason, 'outside_max_angle')

    def test_linear_cde_is_applied(self) -> None:
        result = project_ftheta_point(Vector3(1.0, 0.0, 10.0), width=300, height=200, principal_point_x=100.0, principal_point_y=50.0, angle_to_pixeldist_poly=[0.0, 50.0], max_angle_rad=1.0, linear_c=2.0, linear_e=0.5)
        self.assertGreater(result.u, 100.0)
        self.assertGreater(result.v, 50.0)

class CalibrationParserTests(unittest.TestCase):

    def test_parser_preserves_reference_poly_but_uses_forward_poly(self) -> None:
        result = parse_camera_calibration(camera_calibration_dict(), camera_name='test')
        self.assertEqual(result.reference_poly, 'PIXELDIST_TO_ANGLE')
        self.assertEqual(result.angle_to_pixeldist_poly, (0.0, 50.0))
        projection = result.project_camera_point(Vector3(0.0, 0.0, 2.0))
        self.assertTrue(projection.valid)

    def test_resolution_scaling_matches_driver_policy(self) -> None:
        result = parse_camera_calibration(camera_calibration_dict(), camera_name='test', source_width=100, source_height=50)
        self.assertEqual(result.width, 100)
        self.assertEqual(result.height, 50)
        self.assertAlmostEqual(result.principal_point_x, 50.0)
        self.assertAlmostEqual(result.principal_point_y, 25.0)
        self.assertEqual(result.angle_to_pixeldist_poly, (0.0, 25.0))

    def test_real_front_wide_calibration_parses_when_available(self) -> None:
        path = ALPASIM_DATA_ROOT / 'test_clip_001' / 'calibration' / 'front_wide.json'
        if not path.is_file():
            self.skipTest('real test_clip_001 calibration is unavailable')
        result = load_camera_calibration(path, camera_name='front_wide')
        self.assertIsInstance(result, FthetaCameraCalibration)
        self.assertEqual(result.width, 1920)
        self.assertEqual(result.height, 1080)
        self.assertEqual(result.logical_id, 'camera_front_wide_120fov')
        self.assertGreater(len(result.angle_to_pixeldist_poly), 1)
if __name__ == '__main__':
    unittest.main(verbosity=2)

def box_quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> dict:
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    return {'x': sr * cp * cy - cr * sp * sy, 'y': cr * sp * cy + sr * cp * sy, 'z': cr * cp * sy - sr * sp * cy, 'w': cr * cp * cy + sr * sp * sy}

def box_actor(*, x=0.0, y=0.0, z=1.0, dimensions=(4.0, 2.0, 2.0), rpy=(0.0, 0.0, 0.0)) -> dict:
    return {'track_id': 'actor-1', 'label_class': 'automobile', 'pose': {'position': {'x': x, 'y': y, 'z': z}, 'orientation': box_quaternion_from_rpy(*rpy)}, 'dimensions': {'x': dimensions[0], 'y': dimensions[1], 'z': dimensions[2]}}

def box_ego_message(*, x=0.0, y=0.0, z=0.0, rpy=(0.0, 0.0, 0.0)) -> dict:
    return {'pose_frame_id': 'map', 'position': {'x': x, 'y': y, 'z': z}, 'orientation': box_quaternion_from_rpy(*rpy)}

class LocalBoxTests(unittest.TestCase):

    def test_box_is_centered_on_pose(self) -> None:
        corners = local_box_corners(Vector3(4.0, 2.0, 2.0))
        self.assertEqual(len(corners), 8)
        self.assertEqual({corner.x for corner in corners}, {-2.0, 2.0})
        self.assertEqual({corner.y for corner in corners}, {-1.0, 1.0})
        self.assertEqual({corner.z for corner in corners}, {-1.0, 1.0})

    def test_non_positive_dimension_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            local_box_corners(Vector3(4.0, 0.0, 2.0))

class ActorBoxTests(unittest.TestCase):

    def test_identity_box_uses_pose_as_center(self) -> None:
        box = build_actor_box3d(box_actor(x=10.0, y=5.0, z=2.0))
        self.assertEqual(len(box.corners_map), 8)
        self.assertEqual({corner.x for corner in box.corners_map}, {8.0, 12.0})
        self.assertEqual({corner.y for corner in box.corners_map}, {4.0, 6.0})
        self.assertEqual({corner.z for corner in box.corners_map}, {1.0, 3.0})

    def test_yaw_rotates_length_axis(self) -> None:
        box = build_actor_box3d(box_actor(rpy=(0.0, 0.0, math.pi / 2.0)))
        xs = [corner.x for corner in box.corners_map]
        ys = [corner.y for corner in box.corners_map]
        self.assertAlmostEqual(min(xs), -1.0, places=7)
        self.assertAlmostEqual(max(xs), 1.0, places=7)
        self.assertAlmostEqual(min(ys), -2.0, places=7)
        self.assertAlmostEqual(max(ys), 2.0, places=7)

    def test_full_quaternion_pitch_is_applied(self) -> None:
        box = build_actor_box3d(box_actor(rpy=(0.0, math.pi / 2.0, 0.0)))
        x_extent = max((c.x for c in box.corners_map)) - min((c.x for c in box.corners_map))
        z_extent = max((c.z for c in box.corners_map)) - min((c.z for c in box.corners_map))
        self.assertAlmostEqual(x_extent, 2.0, places=7)
        self.assertAlmostEqual(z_extent, 4.0, places=7)

class MapRigTransformTests(unittest.TestCase):

    def test_map_rig_round_trip(self) -> None:
        ego = Pose3D(position=Vector3(10.0, 2.0, 1.0), orientation=Quaternion(0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)))
        original = Vector3(12.0, 6.0, 3.0)
        rig = map_point_to_rig(original, ego)
        recovered = rig_point_to_map(rig, ego)
        self.assertAlmostEqual(recovered.x, original.x, places=7)
        self.assertAlmostEqual(recovered.y, original.y, places=7)
        self.assertAlmostEqual(recovered.z, original.z, places=7)

    def test_actor_box_corners_in_identity_rig(self) -> None:
        corners = actor_box_corners_in_rig(box_actor(x=5.0, y=0.0, z=1.0), recorded_ego_message=box_ego_message())
        self.assertEqual({corner.x for corner in corners}, {3.0, 7.0})
        self.assertEqual({corner.y for corner in corners}, {-1.0, 1.0})
        self.assertEqual({corner.z for corner in corners}, {0.0, 2.0})

    def test_invalid_ego_pose_frame_is_rejected(self) -> None:
        message = box_ego_message()
        message['pose_frame_id'] = 'base_link'
        with self.assertRaises(ValueError):
            actor_box_corners_in_rig(box_actor(), recorded_ego_message=message)
if __name__ == '__main__':
    unittest.main(verbosity=2)

def adaptive_calibration() -> FthetaCameraCalibration:
    return FthetaCameraCalibration(camera_name='test', logical_id='camera_test', width=2000, height=1200, principal_point_x=1000.0, principal_point_y=600.0, angle_to_pixeldist_poly=(0.0, 600.0, 0.0, 120.0), pixeldist_to_angle_poly=(0.0, 1.0 / 600.0), reference_poly='ANGLE_TO_PIXELDIST', max_angle_rad=1.5, linear_c=1.0, linear_d=0.0, linear_e=0.0, rig_to_camera_translation=Vector3(0.0, 0.0, 0.0), rig_to_camera_rotation=Quaternion(0.0, 0.0, 0.0, 1.0))

class DistanceTests(unittest.TestCase):

    def test_distance_to_horizontal_segment(self) -> None:
        self.assertAlmostEqual(point_to_segment_distance_px(5.0, 3.0, 0.0, 0.0, 10.0, 0.0), 3.0)

    def test_distance_to_degenerate_segment(self) -> None:
        self.assertAlmostEqual(point_to_segment_distance_px(3.0, 4.0, 0.0, 0.0, 0.0, 0.0), 5.0)

class AdaptiveSamplingTests(unittest.TestCase):

    def test_nearly_straight_projection_stops_immediately(self) -> None:
        result = sample_projected_edge_adaptive(Vector3(-0.1, 0.0, 10.0), Vector3(0.1, 0.0, 10.0), adaptive_calibration(), maximum_chord_error_px=1.0, maximum_depth=8)
        self.assertEqual(len(result.projections), 3)
        self.assertEqual(result.maximum_depth_reached, 0)
        self.assertFalse(result.stopped_by_depth_limit)

    def test_tighter_error_produces_at_least_as_many_samples(self) -> None:
        first = Vector3(2.0, -3.0, 2.0)
        second = Vector3(8.0, 4.0, 1.0)
        loose = sample_projected_edge_adaptive(first, second, adaptive_calibration(), maximum_chord_error_px=2.0, maximum_depth=10)
        tight = sample_projected_edge_adaptive(first, second, adaptive_calibration(), maximum_chord_error_px=0.25, maximum_depth=10)
        self.assertGreaterEqual(len(tight.projections), len(loose.projections))
        self.assertGreater(len(tight.projections), 3)

    def test_endpoints_are_preserved(self) -> None:
        first = Vector3(2.0, -3.0, 2.0)
        second = Vector3(8.0, 4.0, 1.0)
        result = sample_projected_edge_adaptive(first, second, adaptive_calibration(), maximum_chord_error_px=0.5, maximum_depth=10)
        self.assertEqual(result.camera_points[0], first)
        self.assertEqual(result.camera_points[-1], second)

    def test_depth_limit_is_reported(self) -> None:
        result = sample_projected_edge_adaptive(Vector3(2.0, -3.0, 2.0), Vector3(8.0, 4.0, 1.0), adaptive_calibration(), maximum_chord_error_px=1e-12, maximum_depth=1)
        self.assertTrue(result.stopped_by_depth_limit)
        self.assertEqual(result.maximum_depth_reached, 1)

    def test_invalid_error_limit_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_projected_edge_adaptive(Vector3(0.0, 0.0, 1.0), Vector3(0.1, 0.0, 1.0), adaptive_calibration(), maximum_chord_error_px=0.0, maximum_depth=4)

    def test_unprojectable_endpoint_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            sample_projected_edge_adaptive(Vector3(0.0, 0.0, -1.0), Vector3(0.1, 0.0, 1.0), adaptive_calibration(), maximum_chord_error_px=1.0, maximum_depth=4)
if __name__ == '__main__':
    unittest.main(verbosity=2)

class ConvexHullTests(unittest.TestCase):

    def test_convex_hull_removes_internal_points(self) -> None:
        result = convex_hull([Point2D(0.0, 0.0), Point2D(2.0, 0.0), Point2D(2.0, 2.0), Point2D(0.0, 2.0), Point2D(1.0, 1.0)])
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result), {Point2D(0.0, 0.0), Point2D(2.0, 0.0), Point2D(2.0, 2.0), Point2D(0.0, 2.0)})

    def test_duplicate_points_are_removed(self) -> None:
        result = convex_hull([Point2D(0.0, 0.0), Point2D(0.0, 0.0), Point2D(1.0, 0.0)])
        self.assertEqual(result, (Point2D(0.0, 0.0), Point2D(1.0, 0.0)))

class PolygonAreaTests(unittest.TestCase):

    def test_rectangle_area(self) -> None:
        self.assertEqual(polygon_area((Point2D(0.0, 0.0), Point2D(4.0, 0.0), Point2D(4.0, 3.0), Point2D(0.0, 3.0))), 12.0)

    def test_degenerate_polygon_area_is_zero(self) -> None:
        self.assertEqual(polygon_area((Point2D(0.0, 0.0),)), 0.0)

class ImageClipTests(unittest.TestCase):

    def test_polygon_fully_inside_is_unchanged_in_area(self) -> None:
        polygon = (Point2D(1.0, 1.0), Point2D(8.0, 1.0), Point2D(8.0, 8.0), Point2D(1.0, 8.0))
        clipped = clip_polygon_to_image(polygon, width=10, height=10)
        self.assertAlmostEqual(polygon_area(clipped), polygon_area(polygon))

    def test_polygon_crossing_right_edge_is_clipped(self) -> None:
        polygon = (Point2D(5.0, 2.0), Point2D(15.0, 2.0), Point2D(15.0, 8.0), Point2D(5.0, 8.0))
        clipped = clip_polygon_to_image(polygon, width=10, height=10)
        self.assertAlmostEqual(polygon_area(clipped), 24.0)
        self.assertTrue(all((point.u <= 9.0 for point in clipped)))

    def test_polygon_fully_outside_becomes_empty(self) -> None:
        polygon = (Point2D(20.0, 2.0), Point2D(30.0, 2.0), Point2D(30.0, 8.0), Point2D(20.0, 8.0))
        self.assertEqual(clip_polygon_to_image(polygon, width=10, height=10), ())

class SummaryTests(unittest.TestCase):

    def test_inside_hull_ratio_is_one(self) -> None:
        result = summarize_projected_hull([Point2D(1.0, 1.0), Point2D(8.0, 1.0), Point2D(8.0, 8.0), Point2D(1.0, 8.0)], width=10, height=10)
        self.assertAlmostEqual(result.inside_image_hull_ratio, 1.0)
        self.assertFalse(result.truncated_by_image)

    def test_partial_hull_has_fractional_ratio(self) -> None:
        result = summarize_projected_hull([Point2D(5.0, 2.0), Point2D(15.0, 2.0), Point2D(15.0, 8.0), Point2D(5.0, 8.0)], width=10, height=10)
        self.assertAlmostEqual(result.projected_hull_area_px, 60.0)
        self.assertAlmostEqual(result.inside_image_hull_area_px, 24.0)
        self.assertAlmostEqual(result.inside_image_hull_ratio, 0.4)
        self.assertTrue(result.truncated_by_image)

    def test_to_dict_contains_point_objects_as_dicts(self) -> None:
        result = summarize_projected_hull([Point2D(0.0, 0.0), Point2D(2.0, 0.0), Point2D(1.0, 2.0)], width=10, height=10).to_dict()
        self.assertIsInstance(result['projected_hull'], tuple)
        self.assertIsInstance(result['projected_hull'][0], dict)
if __name__ == '__main__':
    unittest.main(verbosity=2)

def fovclip_theta(point: Vector3) -> float:
    return math.atan2(math.hypot(point.x, point.y), point.z)

class QuadraticIntervalTests(unittest.TestCase):

    def test_inside_between_two_roots(self) -> None:
        result = quadratic_nonpositive_interval(1.0, -1.0, 0.16)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result.start_ratio, 0.2)
        self.assertAlmostEqual(result.end_ratio, 0.8)

    def test_always_inside(self) -> None:
        self.assertEqual(quadratic_nonpositive_interval(0.0, 0.0, -1.0), type(quadratic_nonpositive_interval(0.0, 0.0, -1.0))(0.0, 1.0))

    def test_always_outside(self) -> None:
        self.assertIsNone(quadratic_nonpositive_interval(0.0, 0.0, 1.0))

class AngularFovClipTests(unittest.TestCase):

    def test_fully_inside_is_unchanged(self) -> None:
        first = Vector3(-0.2, 0.0, 2.0)
        second = Vector3(0.2, 0.0, 2.0)
        self.assertEqual(clip_segment_to_angular_fov(first, second, max_angle_rad=0.5), (first, second))

    def test_fully_outside_is_removed(self) -> None:
        result = clip_segment_to_angular_fov(Vector3(10.0, 0.0, 1.0), Vector3(12.0, 0.0, 1.0), max_angle_rad=0.5)
        self.assertIsNone(result)

    def test_one_endpoint_outside_is_clipped_to_boundary(self) -> None:
        maximum = 0.5
        result = clip_segment_to_angular_fov(Vector3(0.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), max_angle_rad=maximum)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(fovclip_theta(result[0]), 0.0)
        self.assertAlmostEqual(fovclip_theta(result[1]), maximum, places=10)

    def test_two_outside_endpoints_can_cross_fov(self) -> None:
        maximum = 0.5
        result = clip_segment_to_angular_fov(Vector3(-4.0, 0.0, 2.0), Vector3(4.0, 0.0, 2.0), max_angle_rad=maximum)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(fovclip_theta(result[0]), maximum, places=10)
        self.assertAlmostEqual(fovclip_theta(result[1]), maximum, places=10)
        self.assertLess(result[0].x, 0.0)
        self.assertGreater(result[1].x, 0.0)

    def test_none_max_angle_returns_original_segment(self) -> None:
        first = Vector3(10.0, 0.0, 1.0)
        second = Vector3(12.0, 0.0, 1.0)
        self.assertEqual(clip_segment_to_angular_fov(first, second, max_angle_rad=None), (first, second))

    def test_nonconvex_angle_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            clip_segment_to_angular_fov(Vector3(0.0, 0.0, 1.0), Vector3(1.0, 0.0, 1.0), max_angle_rad=math.pi / 2.0)
if __name__ == '__main__':
    unittest.main(verbosity=2)

def image_calibration(*, width=200, height=100, max_angle=1.4) -> FthetaCameraCalibration:
    return FthetaCameraCalibration(camera_name='test', logical_id='camera_test', width=width, height=height, principal_point_x=width / 2.0, principal_point_y=height / 2.0, angle_to_pixeldist_poly=(0.0, 50.0), pixeldist_to_angle_poly=(0.0, 0.02), reference_poly='ANGLE_TO_PIXELDIST', max_angle_rad=max_angle, linear_c=1.0, linear_d=0.0, linear_e=0.0, rig_to_camera_translation=Vector3(0.0, 0.0, 0.0), rig_to_camera_rotation=Quaternion(0.0, 0.0, 0.0, 1.0))

def image_camera_box(*, center=(0.0, 0.0, 10.0), size=(2.0, 2.0, 2.0)) -> tuple[Vector3, ...]:
    cx, cy, cz = center
    sx, sy, sz = size
    return tuple((Vector3(cx + dx * sx / 2.0, cy + dy * sy / 2.0, cz + dz * sz / 2.0) for dz in (-1.0, 1.0) for dy in (-1.0, 1.0) for dx in (-1.0, 1.0)))

class EdgeTests(unittest.TestCase):

    def test_twelve_unique_box_edges(self) -> None:
        self.assertEqual(len(BOX_EDGE_INDEX_PAIRS), 12)
        self.assertEqual(len(set(BOX_EDGE_INDEX_PAIRS)), 12)
        self.assertTrue(all((0 <= a < 8 and 0 <= b < 8 for a, b in BOX_EDGE_INDEX_PAIRS)))

    def test_segment_fully_behind_is_removed(self) -> None:
        result = clip_segment_to_positive_z(Vector3(0.0, 0.0, -2.0), Vector3(1.0, 0.0, -1.0), near_plane_m=0.1)
        self.assertIsNone(result)

    def test_segment_crossing_near_plane_is_clipped(self) -> None:
        result = clip_segment_to_positive_z(Vector3(0.0, 0.0, -1.0), Vector3(2.0, 0.0, 1.0), near_plane_m=0.1)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result[0].z, 0.1)
        self.assertAlmostEqual(result[1].z, 1.0)

    def test_edge_sampling_count_for_front_box(self) -> None:
        samples = sample_box_edges_camera(image_camera_box(), samples_per_edge=5)
        self.assertEqual(len(samples), 12 * 5)
        self.assertTrue(all((sample.z > 0.0 for sample in samples)))

class ProjectionSummaryTests(unittest.TestCase):

    def test_centered_front_box_projects_inside_image(self) -> None:
        result = summarize_camera_box_projection(image_camera_box(), calibration=image_calibration(), track_id='1', actor_class='automobile', samples_per_edge=9)
        self.assertTrue(result.projection_valid)
        self.assertIsNotNone(result.projected_bbox)
        self.assertIsNotNone(result.clipped_bbox)
        self.assertAlmostEqual(result.inside_image_ratio, 1.0)
        self.assertFalse(result.truncated)
        self.assertGreater(result.projected_area_px, 0.0)
        self.assertGreater(result.projected_height_px, 0.0)
        self.assertGreater(result.minimum_depth_m or 0.0, 0.0)

    def test_box_behind_camera_is_invalid(self) -> None:
        result = summarize_camera_box_projection(image_camera_box(center=(0.0, 0.0, -10.0)), calibration=image_calibration(), track_id='1', actor_class='automobile')
        self.assertFalse(result.projection_valid)
        self.assertEqual(result.failure_reason, 'box_behind_near_plane')
        self.assertEqual(result.camera_sample_count, 0)

    def test_partially_outside_box_is_truncated(self) -> None:
        result = summarize_camera_box_projection(image_camera_box(center=(60.0, 0.0, 10.0), size=(40.0, 4.0, 4.0)), calibration=image_calibration(width=140, height=100, max_angle=1.5), track_id='1', actor_class='automobile', samples_per_edge=21)
        self.assertTrue(result.projection_valid)
        self.assertTrue(result.truncated)
        self.assertGreater(result.inside_image_ratio, 0.0)
        self.assertLess(result.inside_image_ratio, 1.0)

    def test_box_outside_fov_is_invalid(self) -> None:
        result = summarize_camera_box_projection(image_camera_box(center=(20.0, 0.0, 1.0)), calibration=image_calibration(width=2000, height=1000, max_angle=0.2), track_id='1', actor_class='automobile')
        self.assertFalse(result.projection_valid)
        self.assertEqual(result.failure_reason, 'box_outside_camera_fov')

    def test_hull_area_is_tighter_than_bbox_area(self) -> None:
        result = summarize_camera_box_projection(image_camera_box(center=(8.0, 4.0, 10.0), size=(8.0, 2.0, 4.0)), calibration=image_calibration(width=400, height=300, max_angle=1.5), track_id='1', actor_class='automobile', samples_per_edge=21)
        self.assertTrue(result.projection_valid)
        self.assertGreater(result.projected_hull_area_px, 0.0)
        self.assertLessEqual(result.projected_hull_area_px, result.projected_area_px)
        self.assertLessEqual(result.inside_image_hull_area_px, result.inside_image_area_px)
        self.assertAlmostEqual(result.inside_image_hull_ratio, 1.0)

    def test_truncated_projection_uses_hull_ratio(self) -> None:
        result = summarize_camera_box_projection(image_camera_box(center=(60.0, 0.0, 10.0), size=(40.0, 4.0, 4.0)), calibration=image_calibration(width=140, height=100, max_angle=1.5), track_id='1', actor_class='automobile', samples_per_edge=21)
        self.assertTrue(result.projection_valid)
        self.assertTrue(result.truncated)
        self.assertGreater(result.inside_image_hull_ratio, 0.0)
        self.assertLess(result.inside_image_hull_ratio, 1.0)
        self.assertGreater(len(result.projected_hull), 2)
        self.assertGreater(len(result.clipped_hull), 2)

    def test_adaptive_sampling_is_default(self) -> None:
        result = summarize_camera_box_projection(image_camera_box(), calibration=image_calibration(), track_id='1', actor_class='automobile')
        self.assertTrue(result.projection_valid)
        self.assertEqual(result.edge_samples_per_edge, 0)
        self.assertLess(result.camera_sample_count, 12 * 9)

    def test_fixed_sampling_remains_explicitly_available(self) -> None:
        result = summarize_camera_box_projection(image_camera_box(), calibration=image_calibration(), track_id='1', actor_class='automobile', samples_per_edge=9)
        self.assertTrue(result.projection_valid)
        self.assertEqual(result.edge_samples_per_edge, 9)
        self.assertEqual(result.camera_sample_count, 12 * 9)

    def test_to_dict_is_json_compatible_shape(self) -> None:
        result = summarize_camera_box_projection(image_camera_box(), calibration=image_calibration(), track_id='1', actor_class='automobile').to_dict()
        self.assertEqual(result['track_id'], '1')
        self.assertIsInstance(result['projected_bbox'], dict)
        self.assertIsInstance(result['projected_hull'], tuple)
        self.assertIsInstance(result['projected_hull'][0], dict)
        self.assertIsInstance(result['projection_valid'], bool)
if __name__ == '__main__':
    unittest.main(verbosity=2)
