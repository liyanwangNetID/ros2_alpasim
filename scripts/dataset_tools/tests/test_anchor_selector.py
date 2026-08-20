#!/usr/bin/env python3
"""Unit tests for anchor_selector.py."""

from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from anchor_selector import (  # noqa: E402
    AnchorEvaluation,
    AnchorSelector,
    AnchorSelectorConfig,
)


@dataclass
class FakeIndex:
    timestamps_ns: tuple[int, ...]


@dataclass
class FakeTimedMessage:
    time_error_ns: int


@dataclass
class FakeGroup:
    mode: str
    maximum_skew_ns: int
    maximum_target_error_ns: int


@dataclass
class FakeSequence:
    groups: tuple[FakeGroup, ...]


class FakeReader:
    def __init__(
        self,
        timestamps_by_camera: dict[str, tuple[int, ...]],
        *,
        start_ns: int = 0,
        end_ns: int = 10_000,
        visual_failures: set[int] | None = None,
        history_failures: set[int] | None = None,
        route_failures: set[int] | None = None,
        actor_failures: set[int] | None = None,
        future_failures: set[int] | None = None,
        visual_mode: str = "exact",
    ) -> None:
        self.clip_id = "test_clip_001"
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.camera_indexes = {
            name: FakeIndex(timestamps)
            for name, timestamps in timestamps_by_camera.items()
        }
        self.visual_failures = visual_failures or set()
        self.history_failures = history_failures or set()
        self.route_failures = route_failures or set()
        self.actor_failures = actor_failures or set()
        self.future_failures = future_failures or set()
        self.visual_mode = visual_mode
        self.calls: list[tuple[str, int]] = []

    def get_multicamera_frames(
        self,
        anchor_ns: int,
        *,
        offsets_ns,
        tolerance_ns,
        prefer_exact,
    ):
        self.calls.append(("visual", anchor_ns))
        if anchor_ns in self.visual_failures:
            return None
        mode = self.visual_mode
        groups = tuple(
            FakeGroup(
                mode=mode,
                maximum_skew_ns=0 if mode == "exact" else 20,
                maximum_target_error_ns=0 if mode == "exact" else 10,
            )
            for _ in offsets_ns
        )
        return FakeSequence(groups=groups)

    def get_ego_history(self, anchor_ns: int, **kwargs):
        self.calls.append(("history", anchor_ns))
        return None if anchor_ns in self.history_failures else [object()]

    def get_navigation_route_at(self, anchor_ns: int, **kwargs):
        self.calls.append(("route", anchor_ns))
        if anchor_ns in self.route_failures:
            return None
        return FakeTimedMessage(time_error_ns=-20)

    def get_actors_at(self, anchor_ns: int, **kwargs):
        self.calls.append(("actors", anchor_ns))
        if anchor_ns in self.actor_failures:
            return None
        return FakeTimedMessage(time_error_ns=10)

    def get_future_ego_trajectory(self, anchor_ns: int, **kwargs):
        self.calls.append(("future", anchor_ns))
        return None if anchor_ns in self.future_failures else [object()]


CAMERAS = (
    "front_wide",
    "front_tele",
    "cross_left",
    "cross_right",
)


def shared_reader(timestamps: tuple[int, ...], **kwargs) -> FakeReader:
    return FakeReader(
        {camera: timestamps for camera in CAMERAS},
        **kwargs,
    )


class ConfigTests(unittest.TestCase):
    def test_default_profile(self) -> None:
        config = AnchorSelectorConfig()
        self.assertEqual(config.frame_offsets_ns, (-400_000_000, -200_000_000, 0))
        self.assertEqual(config.ego_history_span_ns, 1_500_000_000)
        self.assertEqual(config.visual_history_span_ns, 400_000_000)
        self.assertEqual(config.minimum_spacing_ns, 500_000_000)

    def test_invalid_offsets_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AnchorSelectorConfig(frame_offsets_ns=(0, -10))
        with self.assertRaises(ValueError):
            AnchorSelectorConfig(frame_offsets_ns=(-10, -5))

    def test_duplicate_camera_names_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AnchorSelectorConfig(camera_names=("front", "front"))


class CandidateSourceTests(unittest.TestCase):
    def test_shared_current_timestamps(self) -> None:
        reader = FakeReader(
            {
                "front_wide": (100, 200, 300),
                "front_tele": (100, 200, 400),
                "cross_left": (50, 100, 200),
                "cross_right": (100, 200, 500),
            }
        )
        selector = AnchorSelector(
            AnchorSelectorConfig(
                ego_history_points=2,
                ego_history_interval_ns=10,
                future_horizon_ns=10,
                minimum_spacing_ns=10,
            )
        )
        self.assertEqual(
            selector.shared_current_timestamps(reader),
            (100, 200),
        )

    def test_missing_camera_rejected(self) -> None:
        reader = FakeReader(
            {
                "front_wide": (100,),
                "front_tele": (100,),
                "cross_left": (100,),
            }
        )
        with self.assertRaises(ValueError):
            AnchorSelector().shared_current_timestamps(reader)

    def test_boundary_filter(self) -> None:
        config = AnchorSelectorConfig(
            frame_offsets_ns=(-40, -20, 0),
            ego_history_points=4,
            ego_history_interval_ns=20,
            future_horizon_ns=100,
            minimum_spacing_ns=50,
        )
        reader = shared_reader((40, 60, 100, 850, 900), start_ns=0, end_ns=950)
        result = AnchorSelector(config).boundary_eligible_timestamps(
            reader,
            (40, 60, 100, 850, 900),
        )
        self.assertEqual(result, (60, 100, 850))


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AnchorSelectorConfig(
            frame_offsets_ns=(-40, -20, 0),
            camera_tolerance_ns=10,
            ego_history_points=4,
            ego_history_interval_ns=20,
            route_maximum_age_ns=20,
            actor_tolerance_ns=10,
            future_horizon_ns=100,
            minimum_spacing_ns=50,
        )
        self.selector = AnchorSelector(self.config)

    def test_valid_anchor(self) -> None:
        reader = shared_reader((100,), start_ns=0, end_ns=500)
        result = self.selector.evaluate_anchor(reader, 100)
        self.assertTrue(result.fully_valid)
        self.assertEqual(result.visual_sequence_mode, "all_exact")
        self.assertEqual(result.visual_group_modes, ("exact", "exact", "exact"))
        self.assertEqual(result.route_time_error_ns, -20)
        self.assertEqual(result.actor_time_error_ns, 10)

    def test_failure_reasons_are_preserved(self) -> None:
        reader = shared_reader(
            (100,),
            start_ns=0,
            end_ns=500,
            visual_failures={100},
            route_failures={100},
        )
        result = self.selector.evaluate_anchor(reader, 100)
        self.assertFalse(result.fully_valid)
        self.assertEqual(
            result.failure_reasons,
            ("multicamera_frames", "navigation_route"),
        )
        self.assertIsNone(result.visual_sequence_mode)
        self.assertIsNone(result.route_time_error_ns)

    def test_approximate_visual_metadata(self) -> None:
        reader = shared_reader(
            (100,),
            start_ns=0,
            end_ns=500,
            visual_mode="approximate",
        )
        result = self.selector.evaluate_anchor(reader, 100)
        self.assertEqual(result.visual_sequence_mode, "all_approximate")
        self.assertEqual(result.maximum_camera_skew_ns, 20)
        self.assertEqual(result.maximum_camera_target_error_ns, 10)

    def test_to_dict_includes_derived_validity(self) -> None:
        reader = shared_reader((100,), start_ns=0, end_ns=500)
        result = self.selector.evaluate_anchor(reader, 100).to_dict()
        self.assertTrue(result["fully_valid"])


class SpacingTests(unittest.TestCase):
    @staticmethod
    def evaluation(timestamp: int, valid: bool = True) -> AnchorEvaluation:
        failures = tuple() if valid else ("multicamera_frames",)
        return AnchorEvaluation(
            clip_id="test_clip_001",
            anchor_ns=timestamp,
            checks={"multicamera_frames": valid},
            failure_reasons=failures,
            visual_sequence_mode="all_exact" if valid else None,
            visual_group_modes=("exact",) if valid else tuple(),
            maximum_camera_skew_ns=0 if valid else None,
            maximum_camera_target_error_ns=0 if valid else None,
            route_time_error_ns=0,
            actor_time_error_ns=0,
        )

    def test_greedy_minimum_spacing(self) -> None:
        selector = AnchorSelector(
            AnchorSelectorConfig(minimum_spacing_ns=500)
        )
        anchors = [
            self.evaluation(1_000),
            self.evaluation(1_200),
            self.evaluation(1_500),
            self.evaluation(1_999),
            self.evaluation(2_000),
        ]
        selected = selector.apply_minimum_spacing(anchors)
        self.assertEqual(
            [anchor.anchor_ns for anchor in selected],
            [1_000, 1_500, 2_000],
        )

    def test_invalid_anchor_is_never_selected(self) -> None:
        selector = AnchorSelector(
            AnchorSelectorConfig(minimum_spacing_ns=100)
        )
        selected = selector.apply_minimum_spacing(
            [self.evaluation(100, False), self.evaluation(200, True)]
        )
        self.assertEqual([anchor.anchor_ns for anchor in selected], [200])


class EndToEndSelectionTests(unittest.TestCase):
    def test_select(self) -> None:
        config = AnchorSelectorConfig(
            frame_offsets_ns=(-40, -20, 0),
            camera_tolerance_ns=10,
            ego_history_points=4,
            ego_history_interval_ns=20,
            route_maximum_age_ns=20,
            actor_tolerance_ns=10,
            future_horizon_ns=100,
            minimum_spacing_ns=100,
        )
        timestamps = (20, 60, 100, 150, 200, 250, 450)
        reader = shared_reader(
            timestamps,
            start_ns=0,
            end_ns=500,
            visual_failures={200},
        )
        result = AnchorSelector(config).select(reader)
        self.assertEqual(result.source_candidate_count, 7)
        self.assertEqual(result.boundary_eligible_count, 5)
        self.assertEqual(result.fully_valid_count, 4)
        self.assertEqual(
            [item.anchor_ns for item in result.selected_anchors],
            [60, 250],
        )
        summary = result.to_summary_dict()
        self.assertEqual(summary["selected_count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
