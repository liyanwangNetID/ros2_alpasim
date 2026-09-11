#!/usr/bin/env python3
"""Tests for the frozen v0.2.2 Meta-action geometry policy."""

from __future__ import annotations

import math
import unittest

from step4.meta_action_rules_v02 import (
    observed_proposal,
    reviewed_in_progress_proposal,
    reviewed_lane_change_to_turn_allowed,
)


def in_progress_geometry(
    *,
    direction: str = "left",
    final_advantage_m: float = -1.0,
    heading_progress_deg: float = 5.0,
):
    return {
        "observed_adjacent_transitions": [],
        "in_progress_candidates": [
            {
                "candidate": True,
                "direction": direction,
                "final_target_advantage_m": final_advantage_m,
                "directional_heading_progress_deg": (
                    heading_progress_deg
                ),
            }
        ],
    }


def lateral_features(
    *,
    ego_heading_deg: float,
    map_heading_deg: float | None,
):
    lateral = {
        "ego_total_yaw_change_rad": math.radians(
            ego_heading_deg
        ),
    }
    if map_heading_deg is not None:
        lateral["map_corridor_heading_change_rad"] = (
            math.radians(map_heading_deg)
        )
    return {"lateral": lateral}


def observed_turn(
    *,
    interpretation: str,
    level: str,
    ego_heading_change_deg: float,
):
    return {
        "observed_adjacent_transitions": [
            {
                "interpretation": interpretation,
                "interpretation_reason": (
                    "adjacent_target_is_not_parallel_"
                    "downstream_corridor"
                ),
                "junction_evidence_level": level,
                "source_heading_residual": {
                    "ego_heading_change_deg": (
                        ego_heading_change_deg
                    ),
                },
            }
        ]
    }


class ObservedProposalTests(unittest.TestCase):
    def test_turn_candidate_is_normalized(self):
        action, reasons = observed_proposal(
            [
                {
                    "interpretation": "turn_left_candidate",
                    "interpretation_reason": "diverging",
                }
            ]
        )

        self.assertEqual(action, "turn_left")
        self.assertEqual(reasons, ["diverging"])

    def test_conflicting_observed_actions_are_unknown(self):
        action, reasons = observed_proposal(
            [
                {"interpretation": "turn_left_candidate"},
                {"interpretation": "keep_direction"},
            ]
        )

        self.assertEqual(action, "unknown")
        self.assertIn(
            "conflicting_observed_geometry_interpretations",
            reasons,
        )


class ReviewedInProgressTests(unittest.TestCase):
    def test_two_degree_residual_allows_lane_change(self):
        action, reasons = reviewed_in_progress_proposal(
            in_progress_geometry(),
            lateral_features(
                ego_heading_deg=4.1,
                map_heading_deg=2.0,
            ),
        )

        self.assertEqual(action, "change_lane_left")
        self.assertIn(
            "absolute_ego_to_map_heading_residual_at_least_2deg",
            reasons,
        )

    def test_small_residual_rejects_lane_change(self):
        action, reasons = reviewed_in_progress_proposal(
            in_progress_geometry(),
            lateral_features(
                ego_heading_deg=8.4,
                map_heading_deg=8.0,
            ),
        )

        self.assertIsNone(action)
        self.assertEqual(reasons, [])

    def test_missing_map_heading_rejects_lane_change(self):
        action, reasons = reviewed_in_progress_proposal(
            in_progress_geometry(),
            lateral_features(
                ego_heading_deg=4.0,
                map_heading_deg=None,
            ),
        )

        self.assertIsNone(action)
        self.assertEqual(reasons, [])

    def test_large_heading_progress_is_rejected(self):
        action, _ = reviewed_in_progress_proposal(
            in_progress_geometry(
                direction="right",
                final_advantage_m=-0.45,
                heading_progress_deg=16.65,
            ),
            lateral_features(
                ego_heading_deg=-5.0,
                map_heading_deg=-1.0,
            ),
        )

        self.assertIsNone(action)

    def test_target_advantage_below_limit_is_rejected(self):
        action, _ = reviewed_in_progress_proposal(
            in_progress_geometry(
                final_advantage_m=-2.14,
                heading_progress_deg=4.24,
            ),
            lateral_features(
                ego_heading_deg=5.0,
                map_heading_deg=1.0,
            ),
        )

        self.assertIsNone(action)

    def test_conflicting_candidate_directions_are_unknown(self):
        geometry = {
            "in_progress_candidates": [
                {
                    "candidate": True,
                    "direction": "left",
                    "final_target_advantage_m": -1.0,
                    "directional_heading_progress_deg": 5.0,
                },
                {
                    "candidate": True,
                    "direction": "right",
                    "final_target_advantage_m": -1.0,
                    "directional_heading_progress_deg": 5.0,
                },
            ]
        }

        action, reasons = reviewed_in_progress_proposal(
            geometry,
            lateral_features(
                ego_heading_deg=5.0,
                map_heading_deg=1.0,
            ),
        )

        self.assertEqual(action, "unknown")
        self.assertIn(
            "conflicting_reviewed_in_progress_directions",
            reasons,
        )


class LaneChangeToTurnGateTests(unittest.TestCase):
    def test_level_a_and_strong_yaw_allow_revision(self):
        allowed, reasons = (
            reviewed_lane_change_to_turn_allowed(
                "change_lane_right",
                "turn_left",
                observed_turn(
                    interpretation="turn_left_candidate",
                    level="A",
                    ego_heading_change_deg=16.39,
                ),
            )
        )

        self.assertTrue(allowed)
        self.assertIn(
            "reviewed_lane_change_to_turn_geometry",
            reasons,
        )

    def test_level_b_and_boundary_yaw_allow_revision(self):
        allowed, _ = reviewed_lane_change_to_turn_allowed(
            "change_lane_left",
            "turn_right",
            observed_turn(
                interpretation="turn_right_candidate",
                level="B",
                ego_heading_change_deg=-8.0,
            ),
        )

        self.assertTrue(allowed)

    def test_level_c_rejects_revision(self):
        allowed, reasons = (
            reviewed_lane_change_to_turn_allowed(
                "change_lane_left",
                "turn_right",
                observed_turn(
                    interpretation="turn_right_candidate",
                    level="C",
                    ego_heading_change_deg=-20.0,
                ),
            )
        )

        self.assertFalse(allowed)
        self.assertIn(
            "requires_junction_level_a_or_b",
            reasons,
        )

    def test_weak_yaw_rejects_revision(self):
        allowed, reasons = (
            reviewed_lane_change_to_turn_allowed(
                "change_lane_right",
                "turn_left",
                observed_turn(
                    interpretation="turn_left_candidate",
                    level="A",
                    ego_heading_change_deg=4.13,
                ),
            )
        )

        self.assertFalse(allowed)
        self.assertIn(
            "requires_post_transition_ego_heading_change_at_least_8deg",
            reasons,
        )

    def test_non_lane_change_source_is_not_restricted(self):
        allowed, reasons = (
            reviewed_lane_change_to_turn_allowed(
                "keep_direction",
                "turn_right",
                observed_turn(
                    interpretation="turn_right_candidate",
                    level="C",
                    ego_heading_change_deg=-3.0,
                ),
            )
        )

        self.assertTrue(allowed)
        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
