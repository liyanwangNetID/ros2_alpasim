#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from natural_corridor_family_guard_v01 import (
    DirectionFamilyObservation,
    direction_family,
    guard_from_observations,
)


def observations(natural, route):
    return tuple(
        DirectionFamilyObservation(horizon, natural, route)
        for horizon in (40.0, 60.0, 80.0)
    )


class DirectionFamilyGuardTests(unittest.TestCase):
    def test_direction_family_threshold(self):
        self.assertEqual(direction_family(math.radians(16), threshold_deg=15), "left")
        self.assertEqual(direction_family(math.radians(-16), threshold_deg=15), "right")
        self.assertEqual(direction_family(math.radians(10), threshold_deg=15), "straight")

    def test_different_stable_families_accept_directional(self):
        result = guard_from_observations(
            raw_relation="left_of_natural",
            natural_successor_lane_id="S",
            route_successor_lane_id="L",
            observations=observations("straight", "left"),
        )
        self.assertEqual(result.relation, "left_of_natural")
        self.assertEqual(result.status, "accepted_directional")

    def test_same_family_becomes_unresolved(self):
        result = guard_from_observations(
            raw_relation="right_of_natural",
            natural_successor_lane_id="N",
            route_successor_lane_id="R",
            observations=observations("right", "right"),
        )
        self.assertEqual(result.relation, "family_guard_unresolved")
        self.assertEqual(result.family_relationship, "same_family_as_natural")

    def test_unstable_family_becomes_unresolved(self):
        values = (
            DirectionFamilyObservation(40.0, "straight", "left"),
            DirectionFamilyObservation(60.0, "straight", "straight"),
            DirectionFamilyObservation(80.0, "straight", "left"),
        )
        result = guard_from_observations(
            raw_relation="left_of_natural",
            natural_successor_lane_id="N",
            route_successor_lane_id="L",
            observations=values,
        )
        self.assertEqual(result.relation, "family_guard_unresolved")
        self.assertEqual(result.family_relationship, "family_unresolved")

    def test_consistent_natural_continuation_is_unchanged(self):
        result = guard_from_observations(
            raw_relation="natural_continuation",
            natural_successor_lane_id="S",
            route_successor_lane_id="S",
            observations=(),
        )
        self.assertEqual(result.relation, "natural_continuation")
        self.assertEqual(result.status, "not_applicable")

    def test_inconsistent_natural_continuation_is_unresolved(self):
        result = guard_from_observations(
            raw_relation="natural_continuation",
            natural_successor_lane_id="N",
            route_successor_lane_id="R",
            observations=(),
        )
        self.assertEqual(result.relation, "family_guard_unresolved")
        self.assertEqual(
            result.reason,
            "inconsistent_natural_continuation_successor_identity",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
