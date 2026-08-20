#!/usr/bin/env python3
"""Unit tests for temporal_index.py."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from temporal_index import (  # noqa: E402
    TemporalIndex,
    TemporalIndexError,
    TemporalMatchError,
    TimestampOrderError,
    synchronize_camera_sequence,
    synchronize_cameras_at,
)


MS = 1_000_000


class TemporalIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = TemporalIndex(
            [100 * MS, 200 * MS, 300 * MS],
            ["a", "b", "c"],
            name="test",
        )

    def test_nearest_exact(self) -> None:
        match = self.index.nearest(200 * MS)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.value, "b")
        self.assertEqual(match.error_ns, 0)

    def test_nearest_tie_prefers_earlier(self) -> None:
        match = self.index.nearest(250 * MS)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.timestamp_ns, 200 * MS)
        self.assertEqual(match.error_ns, -50 * MS)

    def test_nearest_respects_tolerance(self) -> None:
        self.assertIsNone(
            self.index.nearest(250 * MS, tolerance_ns=49 * MS)
        )
        self.assertIsNotNone(
            self.index.nearest(250 * MS, tolerance_ns=50 * MS)
        )

    def test_at_or_before(self) -> None:
        match = self.index.at_or_before(250 * MS)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.timestamp_ns, 200 * MS)
        self.assertEqual(match.error_ns, -50 * MS)

    def test_at_or_after(self) -> None:
        match = self.index.at_or_after(250 * MS)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.timestamp_ns, 300 * MS)
        self.assertEqual(match.error_ns, 50 * MS)

    def test_lookup_outside_range(self) -> None:
        self.assertIsNone(self.index.at_or_before(50 * MS))
        self.assertIsNone(self.index.at_or_after(350 * MS))

    def test_require_nearest_raises(self) -> None:
        with self.assertRaises(TemporalMatchError):
            self.index.require_nearest(500 * MS, tolerance_ns=50 * MS)

    def test_query_offsets(self) -> None:
        matches = self.index.query_offsets(
            300 * MS,
            [-200 * MS, -100 * MS, 0],
            tolerance_ns=0,
        )
        self.assertIsNotNone(matches)
        assert matches is not None
        self.assertEqual([item.value for item in matches], ["a", "b", "c"])

    def test_query_offsets_rejects_reused_frame(self) -> None:
        matches = self.index.query_offsets(
            200 * MS,
            [-10 * MS, 10 * MS],
            tolerance_ns=100 * MS,
            require_unique=True,
        )
        self.assertIsNone(matches)

    def test_empty_index(self) -> None:
        empty = TemporalIndex([], name="empty")
        self.assertEqual(len(empty), 0)
        self.assertIsNone(empty.nearest(0))
        self.assertIsNone(empty.at_or_before(0))
        self.assertIsNone(empty.at_or_after(0))

    def test_duplicate_timestamp_rejected(self) -> None:
        with self.assertRaises(TimestampOrderError):
            TemporalIndex([100, 100])

    def test_out_of_order_timestamp_rejected(self) -> None:
        with self.assertRaises(TimestampOrderError):
            TemporalIndex([200, 100])

    def test_value_length_mismatch_rejected(self) -> None:
        with self.assertRaises(TemporalIndexError):
            TemporalIndex([100, 200], ["only_one"])

    def test_non_integer_timestamp_rejected(self) -> None:
        with self.assertRaises(TypeError):
            TemporalIndex([100, 200.0])

    def test_negative_tolerance_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.index.nearest(100, tolerance_ns=-1)


class CameraSynchronizationTests(unittest.TestCase):
    def test_exact_shared_timestamp(self) -> None:
        indexes = {
            "front": TemporalIndex([100 * MS, 200 * MS], ["f1", "f2"]),
            "left": TemporalIndex([100 * MS, 200 * MS], ["l1", "l2"]),
            "right": TemporalIndex([100 * MS, 200 * MS], ["r1", "r2"]),
        }
        group = synchronize_cameras_at(
            indexes,
            202 * MS,
            tolerance_ns=50 * MS,
            prefer_exact=True,
        )
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group.mode, "exact")
        self.assertEqual(group.maximum_skew_ns, 0)
        self.assertEqual(group.maximum_target_error_ns, 2 * MS)
        self.assertEqual(
            {match.timestamp_ns for match in group.matches.values()},
            {200 * MS},
        )

    def test_approximate_fallback(self) -> None:
        indexes = {
            "front": TemporalIndex([200 * MS], ["front"]),
            "left": TemporalIndex([210 * MS], ["left"]),
            "right": TemporalIndex([190 * MS], ["right"]),
        }
        group = synchronize_cameras_at(
            indexes,
            200 * MS,
            tolerance_ns=50 * MS,
            prefer_exact=True,
        )
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group.mode, "approximate")
        self.assertEqual(group.maximum_skew_ns, 20 * MS)
        self.assertEqual(group.maximum_target_error_ns, 10 * MS)

    def test_approximate_rejects_target_error(self) -> None:
        indexes = {
            "front": TemporalIndex([200 * MS]),
            "left": TemporalIndex([260 * MS]),
            "right": TemporalIndex([200 * MS]),
        }
        group = synchronize_cameras_at(
            indexes,
            200 * MS,
            tolerance_ns=50 * MS,
        )
        self.assertIsNone(group)

    def test_approximate_rejects_camera_skew(self) -> None:
        indexes = {
            "front": TemporalIndex([160 * MS]),
            "left": TemporalIndex([240 * MS]),
            "right": TemporalIndex([200 * MS]),
        }
        group = synchronize_cameras_at(
            indexes,
            200 * MS,
            tolerance_ns=50 * MS,
        )
        self.assertIsNone(group)

    def test_exact_prefers_shared_time_nearest_target(self) -> None:
        indexes = {
            "front": TemporalIndex([100 * MS, 200 * MS, 300 * MS]),
            "left": TemporalIndex([100 * MS, 200 * MS, 300 * MS]),
        }
        group = synchronize_cameras_at(
            indexes,
            260 * MS,
            tolerance_ns=100 * MS,
        )
        self.assertIsNotNone(group)
        assert group is not None
        self.assertEqual(group.mode, "exact")
        self.assertEqual(
            next(iter(group.matches.values())).timestamp_ns,
            300 * MS,
        )

    def test_camera_sequence(self) -> None:
        timestamps = [100 * MS, 300 * MS, 500 * MS]
        indexes = {
            "front": TemporalIndex(timestamps, ["f1", "f2", "f3"]),
            "tele": TemporalIndex(timestamps, ["t1", "t2", "t3"]),
            "left": TemporalIndex(timestamps, ["l1", "l2", "l3"]),
            "right": TemporalIndex(timestamps, ["r1", "r2", "r3"]),
        }
        sequence = synchronize_camera_sequence(
            indexes,
            500 * MS,
            [-400 * MS, -200 * MS, 0],
            tolerance_ns=50 * MS,
        )
        self.assertIsNotNone(sequence)
        assert sequence is not None
        self.assertEqual(len(sequence.groups), 3)
        self.assertEqual(
            [group.target_ns for group in sequence.groups],
            [100 * MS, 300 * MS, 500 * MS],
        )
        self.assertTrue(all(group.mode == "exact" for group in sequence.groups))

    def test_camera_sequence_rejects_reused_frame(self) -> None:
        indexes = {
            "front": TemporalIndex([100 * MS]),
            "left": TemporalIndex([100 * MS]),
        }
        sequence = synchronize_camera_sequence(
            indexes,
            120 * MS,
            [-20 * MS, 0],
            tolerance_ns=50 * MS,
            require_unique_per_camera=True,
        )
        self.assertIsNone(sequence)

    def test_empty_camera_mapping_rejected(self) -> None:
        with self.assertRaises(ValueError):
            synchronize_cameras_at({}, 0, tolerance_ns=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
