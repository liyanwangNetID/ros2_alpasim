#!/usr/bin/env python3
"""Tests for Step 7D.4 Actor box image projection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


from step7.actor_box_image_projection_v01 import (  # noqa: E402
    BOX_EDGE_INDEX_PAIRS,
    clip_segment_to_positive_z,
    sample_box_edges_camera,
    summarize_camera_box_projection,
)
from step7.camera_projection_v01 import (  # noqa: E402
    FthetaCameraCalibration,
    Quaternion,
    Vector3,
)


def calibration(*, width=200, height=100, max_angle=1.4) -> FthetaCameraCalibration:
    return FthetaCameraCalibration(
        camera_name="test",
        logical_id="camera_test",
        width=width,
        height=height,
        principal_point_x=width / 2.0,
        principal_point_y=height / 2.0,
        angle_to_pixeldist_poly=(0.0, 50.0),
        pixeldist_to_angle_poly=(0.0, 0.02),
        reference_poly="ANGLE_TO_PIXELDIST",
        max_angle_rad=max_angle,
        linear_c=1.0,
        linear_d=0.0,
        linear_e=0.0,
        rig_to_camera_translation=Vector3(0.0, 0.0, 0.0),
        rig_to_camera_rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
    )


def camera_box(*, center=(0.0, 0.0, 10.0), size=(2.0, 2.0, 2.0)) -> tuple[Vector3, ...]:
    cx, cy, cz = center
    sx, sy, sz = size
    return tuple(
        Vector3(cx + dx * sx / 2.0, cy + dy * sy / 2.0, cz + dz * sz / 2.0)
        for dz in (-1.0, 1.0)
        for dy in (-1.0, 1.0)
        for dx in (-1.0, 1.0)
    )


class EdgeTests(unittest.TestCase):
    def test_twelve_unique_box_edges(self) -> None:
        self.assertEqual(len(BOX_EDGE_INDEX_PAIRS), 12)
        self.assertEqual(len(set(BOX_EDGE_INDEX_PAIRS)), 12)
        self.assertTrue(all(0 <= a < 8 and 0 <= b < 8 for a, b in BOX_EDGE_INDEX_PAIRS))

    def test_segment_fully_behind_is_removed(self) -> None:
        result = clip_segment_to_positive_z(
            Vector3(0.0, 0.0, -2.0),
            Vector3(1.0, 0.0, -1.0),
            near_plane_m=0.1,
        )
        self.assertIsNone(result)

    def test_segment_crossing_near_plane_is_clipped(self) -> None:
        result = clip_segment_to_positive_z(
            Vector3(0.0, 0.0, -1.0),
            Vector3(2.0, 0.0, 1.0),
            near_plane_m=0.1,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(result[0].z, 0.1)
        self.assertAlmostEqual(result[1].z, 1.0)

    def test_edge_sampling_count_for_front_box(self) -> None:
        samples = sample_box_edges_camera(camera_box(), samples_per_edge=5)
        self.assertEqual(len(samples), 12 * 5)
        self.assertTrue(all(sample.z > 0.0 for sample in samples))


class ProjectionSummaryTests(unittest.TestCase):
    def test_centered_front_box_projects_inside_image(self) -> None:
        result = summarize_camera_box_projection(
            camera_box(),
            calibration=calibration(),
            track_id="1",
            actor_class="automobile",
            samples_per_edge=9,
        )
        self.assertTrue(result.projection_valid)
        self.assertIsNotNone(result.projected_bbox)
        self.assertIsNotNone(result.clipped_bbox)
        self.assertAlmostEqual(result.inside_image_ratio, 1.0)
        self.assertFalse(result.truncated)
        self.assertGreater(result.projected_area_px, 0.0)
        self.assertGreater(result.projected_height_px, 0.0)
        self.assertGreater(result.minimum_depth_m or 0.0, 0.0)

    def test_box_behind_camera_is_invalid(self) -> None:
        result = summarize_camera_box_projection(
            camera_box(center=(0.0, 0.0, -10.0)),
            calibration=calibration(),
            track_id="1",
            actor_class="automobile",
        )
        self.assertFalse(result.projection_valid)
        self.assertEqual(result.failure_reason, "box_behind_near_plane")
        self.assertEqual(result.camera_sample_count, 0)

    def test_partially_outside_box_is_truncated(self) -> None:
        result = summarize_camera_box_projection(
            camera_box(center=(60.0, 0.0, 10.0), size=(40.0, 4.0, 4.0)),
            calibration=calibration(width=140, height=100, max_angle=1.5),
            track_id="1",
            actor_class="automobile",
            samples_per_edge=21,
        )
        self.assertTrue(result.projection_valid)
        self.assertTrue(result.truncated)
        self.assertGreater(result.inside_image_ratio, 0.0)
        self.assertLess(result.inside_image_ratio, 1.0)

    def test_box_outside_fov_is_invalid(self) -> None:
        result = summarize_camera_box_projection(
            camera_box(center=(20.0, 0.0, 1.0)),
            calibration=calibration(width=2000, height=1000, max_angle=0.2),
            track_id="1",
            actor_class="automobile",
        )
        self.assertFalse(result.projection_valid)
        self.assertEqual(result.failure_reason, "box_outside_camera_fov")

    def test_hull_area_is_tighter_than_bbox_area(self) -> None:
        result = summarize_camera_box_projection(
            camera_box(center=(8.0, 4.0, 10.0), size=(8.0, 2.0, 4.0)),
            calibration=calibration(width=400, height=300, max_angle=1.5),
            track_id="1",
            actor_class="automobile",
            samples_per_edge=21,
        )
        self.assertTrue(result.projection_valid)
        self.assertGreater(result.projected_hull_area_px, 0.0)
        self.assertLessEqual(result.projected_hull_area_px, result.projected_area_px)
        self.assertLessEqual(result.inside_image_hull_area_px, result.inside_image_area_px)
        self.assertAlmostEqual(result.inside_image_hull_ratio, 1.0)

    def test_truncated_projection_uses_hull_ratio(self) -> None:
        result = summarize_camera_box_projection(
            camera_box(center=(60.0, 0.0, 10.0), size=(40.0, 4.0, 4.0)),
            calibration=calibration(width=140, height=100, max_angle=1.5),
            track_id="1",
            actor_class="automobile",
            samples_per_edge=21,
        )
        self.assertTrue(result.projection_valid)
        self.assertTrue(result.truncated)
        self.assertGreater(result.inside_image_hull_ratio, 0.0)
        self.assertLess(result.inside_image_hull_ratio, 1.0)
        self.assertGreater(len(result.projected_hull), 2)
        self.assertGreater(len(result.clipped_hull), 2)

    def test_adaptive_sampling_is_default(self) -> None:
        result = summarize_camera_box_projection(
            camera_box(), calibration=calibration(),
            track_id="1", actor_class="automobile",
        )
        self.assertTrue(result.projection_valid)
        self.assertEqual(result.edge_samples_per_edge, 0)
        self.assertLess(result.camera_sample_count, 12 * 9)

    def test_fixed_sampling_remains_explicitly_available(self) -> None:
        result = summarize_camera_box_projection(
            camera_box(), calibration=calibration(),
            track_id="1", actor_class="automobile", samples_per_edge=9,
        )
        self.assertTrue(result.projection_valid)
        self.assertEqual(result.edge_samples_per_edge, 9)
        self.assertEqual(result.camera_sample_count, 12 * 9)

    def test_to_dict_is_json_compatible_shape(self) -> None:
        result = summarize_camera_box_projection(
            camera_box(),
            calibration=calibration(),
            track_id="1",
            actor_class="automobile",
        ).to_dict()
        self.assertEqual(result["track_id"], "1")
        self.assertIsInstance(result["projected_bbox"], dict)
        self.assertIsInstance(result["projected_hull"], tuple)
        self.assertIsInstance(result["projected_hull"][0], dict)
        self.assertIsInstance(result["projection_valid"], bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
