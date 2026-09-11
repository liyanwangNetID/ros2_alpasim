#!/usr/bin/env python3
"""Synthetic and real-clip tests for clip_reader.py."""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path



from project_paths import ALPASIM_DATA_ROOT

from step2.clip_reader import (  # noqa: E402
    CAMERA_NAMES,
    DrivingClipReader,
    stamp_mapping_to_ns,
)


NS = 1_000_000_000


def stamp(value_ns: int) -> dict[str, int]:
    return {
        "sec": value_ns // NS,
        "nanosec": value_ns % NS,
    }


def pose(x: float, y: float, yaw: float) -> dict:
    return {
        "position": {"x": x, "y": y, "z": 0.0},
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": math.sin(yaw / 2.0),
            "w": math.cos(yaw / 2.0),
        },
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def create_synthetic_clip(root: Path) -> Path:
    clip = root / "test_clip_001"
    clip.mkdir(parents=True)
    write_json(
        clip / "metadata.json",
        {"dataset_format_version": "0.2-batch", "status": "complete"},
    )
    write_json(
        clip / "validation.json",
        {
            "valid": True,
            "first_sim_time_ns": 0,
            "last_sim_time_ns": 5 * NS,
        },
    )

    camera_stamps = [
        1_600_000_000,
        1_800_000_000,
        2_000_000_000,
    ]
    for camera_name in CAMERA_NAMES:
        camera_dir = clip / "cameras" / camera_name
        rows = []
        for index, timestamp in enumerate(camera_stamps):
            image_name = f"{index:06d}.jpg"
            image_path = camera_dir / image_name
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"jpeg")
            rows.append(
                {
                    "frame_index": index,
                    "stamp_ns": timestamp,
                    "frame_id": camera_name + "_optical",
                    "width": 854,
                    "height": 480,
                    "encoding": "rgb8",
                    "image_path": image_name,
                }
            )
        write_jsonl(camera_dir / "timestamps.jsonl", rows)

    executed = []
    for index in range(51):
        timestamp = index * 100_000_000
        executed.append(
            {
                "stamp": stamp(timestamp),
                "pose": pose(index * 0.1, 0.0, 0.0),
                "linear_acceleration": {"x": 0.1, "y": 0.0, "z": 0.0},
                "speed": 1.0,
                "yaw_rate": 0.0,
            }
        )
    write_jsonl(clip / "ego" / "executed_path_points.jsonl", executed)

    recorded_ego_rows = []
    for index in range(51):
        timestamp = index * 100_000_000
        ego_pose = pose(index * 0.1, 0.0, 0.0)
        recorded_ego_rows.append(
            {
                "topic": "/alpasim/ego_state",
                "message": {
                    "stamp": stamp(timestamp),
                    "pose_frame_id": "map",
                    "child_frame_id": "base_link",
                    "dynamics_frame_id": "base_link",
                    "position": ego_pose["position"],
                    "orientation": ego_pose["orientation"],
                    "linear_velocity": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "linear_acceleration": {"x": 0.1, "y": 0.0, "z": 0.0},
                    "angular_acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "speed": 1.0,
                },
            }
        )
    write_jsonl(clip / "ego" / "ego_state.jsonl", recorded_ego_rows)

    route_rows = []
    actor_rows = []
    for index in range(51):
        timestamp = index * 100_000_000
        route_rows.append(
            {
                "topic": "/alpasim/route/model_input",
                "message": {
                    "reference_stamp": stamp(timestamp),
                    "frame_id": "base_link",
                    "points": [],
                },
            }
        )
        actor_rows.append(
            {
                "topic": "/alpasim/actors/current",
                "message": {
                    "stamp": stamp(timestamp),
                    "pose_frame_id": "map",
                    "actors": [],
                },
            }
        )
    write_jsonl(
        clip / "route" / "navigation_route_local.jsonl",
        route_rows,
    )
    write_jsonl(clip / "actors" / "current.jsonl", actor_rows)

    gt_points = []
    for index in range(51):
        timestamp = index * 100_000_000
        gt_points.append(
            {
                "stamp": stamp(timestamp),
                "pose": pose(index * 0.1, 0.0, 0.0),
                "speed": 1.0,
                "linear_acceleration": {"x": 0.0, "y": 0.0, "z": 0.0},
                "yaw_rate": 0.0,
            }
        )
    write_json(
        clip / "ego" / "complete_recording_ground_truth.json",
        {"trajectory": {"points": gt_points}},
    )
    write_json(clip / "map" / "vector_map.json", {"lanes": []})
    return clip


class StampTests(unittest.TestCase):
    def test_stamp_mapping_to_ns(self) -> None:
        self.assertEqual(
            stamp_mapping_to_ns({"sec": 12, "nanosec": 34}),
            12 * NS + 34,
        )


class SyntheticClipReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = Path(tempfile.mkdtemp())
        self.clip = create_synthetic_clip(self.temporary)
        self.reader = DrivingClipReader(self.clip)

    def tearDown(self) -> None:
        shutil.rmtree(self.temporary)

    def test_basic_metadata(self) -> None:
        self.assertEqual(self.reader.clip_id, "test_clip_001")
        self.assertEqual(self.reader.start_ns, 0)
        self.assertEqual(self.reader.end_ns, 5 * NS)

    def test_multicamera_three_frame_sequence(self) -> None:
        sequence = self.reader.get_multicamera_frames(2 * NS)
        self.assertIsNotNone(sequence)
        assert sequence is not None
        self.assertEqual(len(sequence.groups), 3)
        self.assertTrue(all(group.mode == "exact" for group in sequence.groups))
        self.assertTrue(
            all(len(group.matches) == 4 for group in sequence.groups)
        )

    def test_ego_history(self) -> None:
        history = self.reader.get_ego_history(2 * NS)
        self.assertIsNotNone(history)
        assert history is not None
        self.assertEqual(len(history), 16)
        self.assertAlmostEqual(history[-1].relative_x, 0.0)
        self.assertAlmostEqual(history[-1].relative_y, 0.0)
        self.assertAlmostEqual(history[-1].relative_time, 0.0)
        self.assertAlmostEqual(history[0].relative_time, -1.5)
        self.assertAlmostEqual(history[0].relative_x, -1.5)

    def test_recorded_ego_state_exact_match(self) -> None:
        recorded = self.reader.get_recorded_ego_state_at_or_before(
            2 * NS
        )
        self.assertIsNotNone(recorded)
        assert recorded is not None
        self.assertEqual(recorded.stamp_ns, 2 * NS)
        self.assertEqual(recorded.time_error_ns, 0)
        self.assertEqual(recorded.message["pose_frame_id"], "map")
        self.assertEqual(recorded.message["speed"], 1.0)

    def test_recorded_ego_state_default_rejects_old_state(self) -> None:
        recorded = self.reader.get_recorded_ego_state_at_or_before(
            2_050_000_000
        )
        self.assertIsNone(recorded)

    def test_recorded_ego_state_accepts_explicit_maximum_age(self) -> None:
        recorded = self.reader.get_recorded_ego_state_at_or_before(
            2_050_000_000,
            maximum_age_ns=50_000_000,
        )
        self.assertIsNotNone(recorded)
        assert recorded is not None
        self.assertEqual(recorded.stamp_ns, 2 * NS)
        self.assertEqual(recorded.time_error_ns, -50_000_000)
        self.assertLessEqual(recorded.stamp_ns, 2_050_000_000)

    def test_recorded_ego_state_never_uses_future_state(self) -> None:
        recorded = self.reader.get_recorded_ego_state_at_or_before(
            50_000_000,
            maximum_age_ns=100_000_000,
        )
        self.assertIsNotNone(recorded)
        assert recorded is not None
        self.assertEqual(recorded.stamp_ns, 0)
        self.assertLessEqual(recorded.stamp_ns, 50_000_000)
        self.assertLessEqual(recorded.time_error_ns, 0)

    def test_recorded_ego_state_rejects_invalid_age(self) -> None:
        with self.assertRaises(ValueError):
            self.reader.get_recorded_ego_state_at_or_before(
                2 * NS,
                maximum_age_ns=-1,
            )
        with self.assertRaises(TypeError):
            self.reader.get_recorded_ego_state_at_or_before(
                2 * NS,
                maximum_age_ns=True,
            )

    def test_recorded_ego_state_reuses_cache(self) -> None:
        self.assertIsNone(self.reader._recorded_ego_index)
        self.assertIsNone(self.reader._recorded_ego_records)
        self.reader.get_recorded_ego_state_at_or_before(2 * NS)
        index = self.reader._recorded_ego_index
        records = self.reader._recorded_ego_records
        self.reader.get_recorded_ego_state_at_or_before(3 * NS)
        self.assertIs(self.reader._recorded_ego_index, index)
        self.assertIs(self.reader._recorded_ego_records, records)

    def test_existing_interpolated_ego_state_behavior_is_preserved(self) -> None:
        interpolated = self.reader.get_ego_state_at(2_050_000_000)
        recorded = self.reader.get_recorded_ego_state_at_or_before(
            2_050_000_000
        )
        self.assertIsNotNone(interpolated)
        assert interpolated is not None
        self.assertEqual(interpolated.stamp_ns, 2_050_000_000)
        self.assertIsNone(recorded)

    def test_route_and_actors(self) -> None:
        route = self.reader.get_navigation_route_at(2 * NS)
        actors = self.reader.get_actors_at(2 * NS)
        self.assertIsNotNone(route)
        self.assertIsNotNone(actors)
        assert route is not None and actors is not None
        self.assertEqual(route.time_error_ns, 0)
        self.assertEqual(actors.time_error_ns, 0)

    def test_actor_snapshots_include_closed_window(self) -> None:
        snapshots = self.reader.get_actor_snapshots(
            2 * NS,
            duration_ns=300_000_000,
        )
        self.assertEqual(
            [item.stamp_ns for item in snapshots],
            [1_700_000_000, 1_800_000_000, 1_900_000_000, 2_000_000_000],
        )
        self.assertEqual(
            [item.time_error_ns for item in snapshots],
            [-300_000_000, -200_000_000, -100_000_000, 0],
        )

    def test_actor_snapshots_never_include_future_data(self) -> None:
        end_ns = 2_050_000_000
        snapshots = self.reader.get_actor_snapshots(
            end_ns,
            duration_ns=250_000_000,
        )
        self.assertEqual(
            [item.stamp_ns for item in snapshots],
            [1_800_000_000, 1_900_000_000, 2_000_000_000],
        )
        self.assertTrue(all(item.stamp_ns <= end_ns for item in snapshots))
        self.assertTrue(all(item.time_error_ns <= 0 for item in snapshots))

    def test_actor_snapshots_zero_duration_requires_exact_time(self) -> None:
        exact = self.reader.get_actor_snapshots(2 * NS, duration_ns=0)
        missing = self.reader.get_actor_snapshots(
            2_050_000_000,
            duration_ns=0,
        )
        self.assertEqual([item.stamp_ns for item in exact], [2 * NS])
        self.assertEqual(missing, ())

    def test_actor_snapshots_reject_invalid_duration(self) -> None:
        with self.assertRaises(ValueError):
            self.reader.get_actor_snapshots(2 * NS, duration_ns=-1)
        with self.assertRaises(TypeError):
            self.reader.get_actor_snapshots(2 * NS, duration_ns=True)

    def test_actor_snapshots_reuse_actor_cache(self) -> None:
        self.assertIsNone(self.reader._actor_index)
        self.assertIsNone(self.reader._actor_records)
        self.reader.get_actor_snapshots(2 * NS, duration_ns=300_000_000)
        actor_index = self.reader._actor_index
        actor_records = self.reader._actor_records
        self.reader.get_actor_snapshots(3 * NS, duration_ns=300_000_000)
        self.assertIs(self.reader._actor_index, actor_index)
        self.assertIs(self.reader._actor_records, actor_records)

    def test_future_trajectory(self) -> None:
        future = self.reader.get_future_ego_trajectory(
            2 * NS,
            horizon_ns=2 * NS,
        )
        self.assertIsNotNone(future)
        assert future is not None
        self.assertEqual(future.end_ns, 4 * NS)
        self.assertGreaterEqual(len(future.points), 20)

    def test_vector_map_is_cached(self) -> None:
        first = self.reader.get_vector_map()
        second = self.reader.get_vector_map()
        self.assertIs(first, second)

    def test_validate_anchor(self) -> None:
        result = self.reader.validate_anchor(
            2 * NS,
            future_horizon_ns=2 * NS,
        )
        self.assertTrue(result.usable)
        self.assertFalse(result.reasons)

    def test_early_anchor_is_not_usable(self) -> None:
        result = self.reader.validate_anchor(
            500_000_000,
            future_horizon_ns=2 * NS,
        )
        self.assertFalse(result.usable)
        self.assertFalse(result.checks["ego_history"])


class RealClipSmokeTest(unittest.TestCase):
    def test_real_clip_001(self) -> None:
        clip = ALPASIM_DATA_ROOT / "test_clip_001"
        if not clip.is_dir():
            self.skipTest("real test_clip_001 is not available")
        reader = DrivingClipReader(clip)
        self.assertTrue(reader.validation["valid"])
        self.assertEqual(set(reader.camera_indexes), set(CAMERA_NAMES))
        anchor_ns = reader.start_ns + 3 * NS
        history = reader.get_ego_history(anchor_ns)
        self.assertIsNotNone(history)
        self.assertEqual(len(history or ()), 16)
        self.assertIsNotNone(reader.get_navigation_route_at(anchor_ns))
        self.assertIsNotNone(reader.get_actors_at(anchor_ns))
        self.assertIsInstance(reader.get_vector_map(), dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
