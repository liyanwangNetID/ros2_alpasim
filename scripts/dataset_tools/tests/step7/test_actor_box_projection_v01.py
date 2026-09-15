#!/usr/bin/env python3
"""Tests for Step 7D Actor oriented-box construction."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


from step7.actor_box_projection_v01 import (  # noqa: E402
    Pose3D,
    actor_box_corners_in_rig,
    build_actor_box3d,
    local_box_corners,
    map_point_to_rig,
    rig_point_to_map,
)
from step7.camera_projection_v01 import Quaternion, Vector3  # noqa: E402


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> dict:
    cr = math.cos(roll / 2.0)
    sr = math.sin(roll / 2.0)
    cp = math.cos(pitch / 2.0)
    sp = math.sin(pitch / 2.0)
    cy = math.cos(yaw / 2.0)
    sy = math.sin(yaw / 2.0)
    return {
        "x": sr * cp * cy - cr * sp * sy,
        "y": cr * sp * cy + sr * cp * sy,
        "z": cr * cp * sy - sr * sp * cy,
        "w": cr * cp * cy + sr * sp * sy,
    }


def actor(*, x=0.0, y=0.0, z=1.0, dimensions=(4.0, 2.0, 2.0), rpy=(0.0, 0.0, 0.0)) -> dict:
    return {
        "track_id": "actor-1",
        "label_class": "automobile",
        "pose": {
            "position": {"x": x, "y": y, "z": z},
            "orientation": quaternion_from_rpy(*rpy),
        },
        "dimensions": {
            "x": dimensions[0],
            "y": dimensions[1],
            "z": dimensions[2],
        },
    }


def ego_message(*, x=0.0, y=0.0, z=0.0, rpy=(0.0, 0.0, 0.0)) -> dict:
    return {
        "pose_frame_id": "map",
        "position": {"x": x, "y": y, "z": z},
        "orientation": quaternion_from_rpy(*rpy),
    }


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
        box = build_actor_box3d(actor(x=10.0, y=5.0, z=2.0))
        self.assertEqual(len(box.corners_map), 8)
        self.assertEqual({corner.x for corner in box.corners_map}, {8.0, 12.0})
        self.assertEqual({corner.y for corner in box.corners_map}, {4.0, 6.0})
        self.assertEqual({corner.z for corner in box.corners_map}, {1.0, 3.0})

    def test_yaw_rotates_length_axis(self) -> None:
        box = build_actor_box3d(actor(rpy=(0.0, 0.0, math.pi / 2.0)))
        xs = [corner.x for corner in box.corners_map]
        ys = [corner.y for corner in box.corners_map]
        self.assertAlmostEqual(min(xs), -1.0, places=7)
        self.assertAlmostEqual(max(xs), 1.0, places=7)
        self.assertAlmostEqual(min(ys), -2.0, places=7)
        self.assertAlmostEqual(max(ys), 2.0, places=7)

    def test_full_quaternion_pitch_is_applied(self) -> None:
        box = build_actor_box3d(actor(rpy=(0.0, math.pi / 2.0, 0.0)))
        x_extent = max(c.x for c in box.corners_map) - min(c.x for c in box.corners_map)
        z_extent = max(c.z for c in box.corners_map) - min(c.z for c in box.corners_map)
        self.assertAlmostEqual(x_extent, 2.0, places=7)
        self.assertAlmostEqual(z_extent, 4.0, places=7)


class MapRigTransformTests(unittest.TestCase):
    def test_map_rig_round_trip(self) -> None:
        ego = Pose3D(
            position=Vector3(10.0, 2.0, 1.0),
            orientation=Quaternion(0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
        )
        original = Vector3(12.0, 6.0, 3.0)
        rig = map_point_to_rig(original, ego)
        recovered = rig_point_to_map(rig, ego)
        self.assertAlmostEqual(recovered.x, original.x, places=7)
        self.assertAlmostEqual(recovered.y, original.y, places=7)
        self.assertAlmostEqual(recovered.z, original.z, places=7)

    def test_actor_box_corners_in_identity_rig(self) -> None:
        corners = actor_box_corners_in_rig(
            actor(x=5.0, y=0.0, z=1.0),
            recorded_ego_message=ego_message(),
        )
        self.assertEqual({corner.x for corner in corners}, {3.0, 7.0})
        self.assertEqual({corner.y for corner in corners}, {-1.0, 1.0})
        self.assertEqual({corner.z for corner in corners}, {0.0, 2.0})

    def test_invalid_ego_pose_frame_is_rejected(self) -> None:
        message = ego_message()
        message["pose_frame_id"] = "base_link"
        with self.assertRaises(ValueError):
            actor_box_corners_in_rig(
                actor(),
                recorded_ego_message=message,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
