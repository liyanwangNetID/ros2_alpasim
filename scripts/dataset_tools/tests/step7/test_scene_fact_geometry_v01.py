#!/usr/bin/env python3
"""Tests for Step 7C Ego-relative Actor geometry."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


from step2.coordinate_utils import Pose2D, yaw_to_quaternion  # noqa: E402
from step7.scene_fact_geometry_v01 import (  # noqa: E402
    classify_geometric_region,
    compute_snapshot_actor_geometries,
    recorded_ego_pose_and_speed,
    compute_actor_geometry,
)


def actor(
    *,
    x: float,
    y: float,
    yaw: float = 0.0,
    velocity_x: float = 0.0,
    velocity_y: float = 0.0,
    speed: float = 0.0,
    track_id: str = "actor-1",
    label_class: str = "automobile",
    is_static: bool = False,
) -> dict:
    qx, qy, qz, qw = yaw_to_quaternion(yaw)
    return {
        "track_id": track_id,
        "label_class": label_class,
        "is_static": is_static,
        "pose": {
            "position": {"x": x, "y": y, "z": 0.0},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
        },
        "linear_velocity": {
            "x": velocity_x,
            "y": velocity_y,
            "z": 0.0,
        },
        "speed": speed,
    }


class GeometricRegionTests(unittest.TestCase):
    def test_step7_region_names(self) -> None:
        self.assertEqual(classify_geometric_region(5.0, 0.0), "front")
        self.assertEqual(classify_geometric_region(5.0, 5.0), "front_left")
        self.assertEqual(classify_geometric_region(5.0, -5.0), "front_right")
        self.assertEqual(classify_geometric_region(-5.0, 0.0), "rear")
        self.assertEqual(classify_geometric_region(0.0, 5.0), "left")
        self.assertEqual(classify_geometric_region(0.0, -5.0), "right")
        self.assertEqual(classify_geometric_region(0.1, 0.1), "overlapping")

    def test_negative_deadband_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_geometric_region(
                1.0,
                1.0,
                longitudinal_deadband_m=-1.0,
            )


class ActorGeometryTests(unittest.TestCase):
    def test_identity_anchor_geometry(self) -> None:
        result = compute_actor_geometry(
            actor(x=3.0, y=4.0, velocity_x=8.0, speed=8.0),
            anchor_ego_pose=Pose2D(0.0, 0.0, 0.0),
            ego_speed_mps=5.0,
        )
        self.assertAlmostEqual(result.relative_x_m, 3.0)
        self.assertAlmostEqual(result.relative_y_m, 4.0)
        self.assertAlmostEqual(result.planar_distance_m, 5.0)
        self.assertAlmostEqual(result.relative_longitudinal_speed_mps, 3.0)
        self.assertAlmostEqual(result.relative_lateral_speed_mps, 0.0)
        self.assertEqual(result.geometric_region, "front_left")

    def test_ninety_degree_anchor_rotates_position_and_velocity(self) -> None:
        result = compute_actor_geometry(
            actor(
                x=10.0,
                y=3.0,
                yaw=math.pi / 2.0,
                velocity_x=0.0,
                velocity_y=4.0,
                speed=4.0,
            ),
            anchor_ego_pose=Pose2D(10.0, 1.0, math.pi / 2.0),
            ego_speed_mps=1.0,
        )
        self.assertAlmostEqual(result.relative_x_m, 2.0, places=7)
        self.assertAlmostEqual(result.relative_y_m, 0.0, places=7)
        self.assertAlmostEqual(result.actor_velocity_x_mps, 4.0, places=7)
        self.assertAlmostEqual(result.actor_velocity_y_mps, 0.0, places=7)
        self.assertAlmostEqual(result.relative_velocity_x_mps, 3.0, places=7)
        self.assertAlmostEqual(result.relative_yaw_rad, 0.0, places=7)
        self.assertEqual(result.geometric_region, "front")

    def test_relative_yaw_wraps(self) -> None:
        result = compute_actor_geometry(
            actor(x=1.0, y=0.0, yaw=math.radians(-179.0)),
            anchor_ego_pose=Pose2D(0.0, 0.0, math.radians(179.0)),
            ego_speed_mps=0.0,
        )
        self.assertAlmostEqual(
            result.relative_yaw_rad,
            math.radians(2.0),
        )

    def test_static_actor_is_preserved(self) -> None:
        result = compute_actor_geometry(
            actor(x=2.0, y=0.0, is_static=True),
            anchor_ego_pose=Pose2D(0.0, 0.0, 0.0),
            ego_speed_mps=0.0,
        )
        self.assertTrue(result.is_static)
        self.assertEqual(result.track_id, "actor-1")
        self.assertEqual(result.actor_class, "automobile")

    def test_to_dict_is_json_compatible_shape(self) -> None:
        result = compute_actor_geometry(
            actor(x=2.0, y=-1.0),
            anchor_ego_pose=Pose2D(0.0, 0.0, 0.0),
            ego_speed_mps=0.0,
        ).to_dict()
        self.assertEqual(result["track_id"], "actor-1")
        self.assertEqual(result["geometric_region"], "front_right")
        self.assertIsInstance(result["planar_distance_m"], float)

    def test_missing_track_id_is_rejected(self) -> None:
        value = actor(x=1.0, y=0.0)
        del value["track_id"]
        with self.assertRaises(ValueError):
            compute_actor_geometry(
                value,
                anchor_ego_pose=Pose2D(0.0, 0.0, 0.0),
                ego_speed_mps=0.0,
            )

    def test_boolean_speed_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            compute_actor_geometry(
                actor(x=1.0, y=0.0),
                anchor_ego_pose=Pose2D(0.0, 0.0, 0.0),
                ego_speed_mps=True,
            )

class SnapshotIntegrationTests(unittest.TestCase):
    @staticmethod
    def ego_message() -> dict:
        qx, qy, qz, qw = yaw_to_quaternion(0.0)
        return {
            "pose_frame_id": "map",
            "dynamics_frame_id": "base_link",
            "position": {"x": 10.0, "y": 2.0, "z": 0.0},
            "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
            "speed": 3.0,
        }

    def test_recorded_ego_message_parsing(self) -> None:
        pose, speed = recorded_ego_pose_and_speed(self.ego_message())
        self.assertEqual(pose, Pose2D(10.0, 2.0, 0.0))
        self.assertEqual(speed, 3.0)

    def test_snapshot_geometry_integration(self) -> None:
        snapshot = {
            "pose_frame_id": "map",
            "dynamics_frame_id": "map",
            "actors": [
                actor(x=15.0, y=2.0, velocity_x=4.0, speed=4.0),
                actor(x=8.0, y=4.0, track_id="actor-2"),
            ],
        }
        result = compute_snapshot_actor_geometries(
            snapshot,
            recorded_ego_message=self.ego_message(),
        )
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].relative_x_m, 5.0)
        self.assertAlmostEqual(result[0].relative_velocity_x_mps, 1.0)
        self.assertEqual(result[0].geometric_region, "front")
        self.assertEqual(result[1].geometric_region, "rear_left")

    def test_invalid_ego_frames_are_rejected(self) -> None:
        message = self.ego_message()
        message["pose_frame_id"] = "base_link"
        with self.assertRaises(ValueError):
            recorded_ego_pose_and_speed(message)

    def test_invalid_actor_frames_are_rejected(self) -> None:
        snapshot = {
            "pose_frame_id": "map",
            "dynamics_frame_id": "base_link",
            "actors": [],
        }
        with self.assertRaises(ValueError):
            compute_snapshot_actor_geometries(
                snapshot,
                recorded_ego_message=self.ego_message(),
            )

    def test_duplicate_track_ids_are_rejected(self) -> None:
        snapshot = {
            "pose_frame_id": "map",
            "dynamics_frame_id": "map",
            "actors": [
                actor(x=1.0, y=0.0),
                actor(x=2.0, y=0.0),
            ],
        }
        with self.assertRaises(ValueError):
            compute_snapshot_actor_geometries(
                snapshot,
                recorded_ego_message=self.ego_message(),
            )

if __name__ == "__main__":
    unittest.main(verbosity=2)
