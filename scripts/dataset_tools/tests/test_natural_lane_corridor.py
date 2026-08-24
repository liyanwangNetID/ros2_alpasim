#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from natural_lane_corridor import (
    NaturalCorridorConfig,
    assess_branch_candidate_reliability,
    build_natural_lane_corridor,
    choose_natural_successor,
    compare_actual_lane_sequence,
    evaluate_branch_candidates,
    recover_boundary_branch_comparisons,
)
from vector_map_reader import VectorMapReader


def polyline(points):
    return {
        "points": [{"x": x, "y": y, "z": 0.0} for x, y in points],
        "headings": [],
    }


def lane(lane_id, points, *, predecessors=(), successors=()):
    # Width is only needed to satisfy VectorMapReader parsing.
    left = [(x, y + 1.0) for x, y in points]
    right = [(x, y - 1.0) for x, y in points]
    return {
        "id": lane_id,
        "centerline": polyline(points),
        "left_boundary": polyline(left),
        "right_boundary": polyline(right),
        "predecessor_ids": list(predecessors),
        "successor_ids": list(successors),
        "left_adjacent_ids": [],
        "right_adjacent_ids": [],
        "road_area_ids": [],
        "traffic_sign_ids": [],
        "wait_line_ids": [],
    }


def vector_map(lanes):
    return VectorMapReader.from_dict(
        {
            "frame_id": "map",
            "map_id": "synthetic",
            "revision": 1,
            "lanes": lanes,
            "road_edges": [],
            "traffic_signs": [],
            "wait_lines": [],
        }
    )


def branch_map():
    return vector_map(
        [
            lane("A", [(0, 0), (10, 0)], successors=("S", "L", "R")),
            lane("S", [(10, 0), (20, 0)], predecessors=("A",), successors=("S2",)),
            lane("S2", [(20, 0), (30, 0)], predecessors=("S",)),
            lane("L", [(10, 0), (15, 1), (20, 5)], predecessors=("A",)),
            lane("R", [(10, 0), (15, -1), (20, -5)], predecessors=("A",)),
        ]
    )


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.map = branch_map()
        self.config = NaturalCorridorConfig(
            branch_evaluation_distance_m=20.0,
            maximum_lookahead_m=50.0,
        )

    def test_straight_is_natural_successor(self):
        chosen, candidates = choose_natural_successor(
            self.map, "A", config=self.config
        )
        self.assertEqual(chosen, "S")
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0].successor_lane_id, "S")

    def test_branch_candidates_have_signed_directions(self):
        candidates = evaluate_branch_candidates(
            self.map, "A", config=self.config
        )
        by_id = {candidate.successor_lane_id: candidate for candidate in candidates}
        self.assertGreater(by_id["L"].signed_heading_change_rad, 0.0)
        self.assertLess(by_id["R"].signed_heading_change_rad, 0.0)
        self.assertAlmostEqual(by_id["S"].signed_heading_change_rad, 0.0)

    def test_reliability_rejects_abnormal_curvature(self):
        candidates = evaluate_branch_candidates(
            self.map, "A", config=self.config
        )
        modified = tuple(
            candidate.__class__(
                successor_lane_id=candidate.successor_lane_id,
                lane_ids=candidate.lane_ids,
                evaluated_distance_m=candidate.evaluated_distance_m,
                signed_heading_change_rad=candidate.signed_heading_change_rad,
                absolute_heading_change_rad=(
                    4.0 if candidate is candidates[0]
                    else candidate.absolute_heading_change_rad
                ),
                score=candidate.score,
            )
            for candidate in candidates
        )
        status, reasons, _ = assess_branch_candidate_reliability(
            modified, config=self.config
        )
        self.assertEqual(status, "unreliable")
        self.assertIn("abnormal_candidate_absolute_heading_change", reasons)

    def test_reliability_rejects_tied_candidates(self):
        tied = vector_map(
            [
                lane("A", [(0, 0), (10, 0)], successors=("B", "C")),
                lane("B", [(10, 0), (20, 0)], predecessors=("A",)),
                lane("C", [(10, 0), (20, 0)], predecessors=("A",)),
            ]
        )
        candidates = evaluate_branch_candidates(tied, "A", config=self.config)
        status, reasons, margin = assess_branch_candidate_reliability(
            candidates, config=self.config
        )
        self.assertEqual(status, "unreliable")
        self.assertEqual(margin, 0.0)
        self.assertIn("insufficient_candidate_score_margin", reasons)

    def test_deterministic_tie_breaking(self):
        tied = vector_map(
            [
                lane("A", [(0, 0), (10, 0)], successors=("B", "C")),
                lane("B", [(10, 0), (20, 0)], predecessors=("A",)),
                lane("C", [(10, 0), (20, 0)], predecessors=("A",)),
            ]
        )
        chosen, _ = choose_natural_successor(tied, "A", config=self.config)
        self.assertEqual(chosen, "B")


class CorridorTests(unittest.TestCase):
    def setUp(self):
        self.map = branch_map()
        self.config = NaturalCorridorConfig(
            branch_evaluation_distance_m=20.0,
            maximum_lookahead_m=50.0,
        )

    def test_corridor_follows_natural_branch(self):
        corridor = build_natural_lane_corridor(
            self.map,
            "A",
            lookahead_distance_m=25.0,
            config=self.config,
        )
        self.assertEqual(corridor.lane_ids, ("A", "S", "S2"))
        self.assertEqual(len(corridor.branch_decisions), 1)
        self.assertAlmostEqual(corridor.total_distance_m, 25.0)
        self.assertEqual(corridor.terminated_reason, "lookahead_reached")

    def test_corridor_starts_at_centerline_projection_offset(self):
        corridor = build_natural_lane_corridor(
            self.map,
            "A",
            lookahead_distance_m=12.0,
            start_arc_length_m=6.0,
            config=self.config,
        )
        self.assertAlmostEqual(corridor.points[0].x, 6.0)
        self.assertAlmostEqual(corridor.points[0].y, 0.0)
        self.assertAlmostEqual(corridor.total_distance_m, 12.0)
        self.assertEqual(corridor.lane_ids, ("A", "S"))

    def test_start_offset_at_lane_end_continues_to_successor(self):
        corridor = build_natural_lane_corridor(
            self.map,
            "A",
            lookahead_distance_m=5.0,
            start_arc_length_m=10.0,
            config=self.config,
        )
        self.assertEqual(corridor.lane_ids, ("A", "S"))
        self.assertAlmostEqual(corridor.total_distance_m, 5.0)

    def test_invalid_start_offset(self):
        with self.assertRaises(ValueError):
            build_natural_lane_corridor(
                self.map,
                "A",
                lookahead_distance_m=5.0,
                start_arc_length_m=11.0,
                config=self.config,
            )

    def test_lookahead_is_capped(self):
        config = NaturalCorridorConfig(maximum_lookahead_m=12.0)
        corridor = build_natural_lane_corridor(
            self.map, "A", lookahead_distance_m=100.0, config=config
        )
        self.assertAlmostEqual(corridor.total_distance_m, 12.0)

    def test_actual_branch_comparison(self):
        corridor = build_natural_lane_corridor(
            self.map, "A", lookahead_distance_m=25.0, config=self.config
        )
        natural = compare_actual_lane_sequence(("A", "S"), corridor)[0]
        left = compare_actual_lane_sequence(("A", "L"), corridor)[0]
        right = compare_actual_lane_sequence(("A", "R"), corridor)[0]
        unobserved = compare_actual_lane_sequence(("L",), corridor)[0]
        self.assertEqual(natural.actual_relation_to_natural, "natural_continuation")
        self.assertTrue(natural.actual_matches_natural)
        self.assertEqual(left.actual_relation_to_natural, "left_of_natural")
        self.assertEqual(right.actual_relation_to_natural, "right_of_natural")
        self.assertEqual(unobserved.actual_relation_to_natural, "not_observed")

    def test_boundary_branch_recovery(self):
        natural = recover_boundary_branch_comparisons(
            self.map, "S", config=self.config
        )[0]
        left = recover_boundary_branch_comparisons(
            self.map, "L", config=self.config
        )[0]
        right = recover_boundary_branch_comparisons(
            self.map, "R", config=self.config
        )[0]
        self.assertEqual(natural.actual_relation_to_natural, "natural_continuation")
        self.assertEqual(left.actual_relation_to_natural, "left_of_natural")
        self.assertEqual(right.actual_relation_to_natural, "right_of_natural")

    def test_cycle_terminates_safely(self):
        cyclic = vector_map(
            [
                lane("A", [(0, 0), (10, 0)], successors=("B",)),
                lane("B", [(10, 0), (20, 0)], predecessors=("A",), successors=("A",)),
            ]
        )
        corridor = build_natural_lane_corridor(
            cyclic, "A", lookahead_distance_m=100.0
        )
        self.assertEqual(corridor.terminated_reason, "cycle_detected")


class ConfigTests(unittest.TestCase):
    def test_invalid_config(self):
        with self.assertRaises(ValueError):
            NaturalCorridorConfig(maximum_lookahead_m=0.0)
        with self.assertRaises(ValueError):
            NaturalCorridorConfig(branch_evaluation_distance_m=-1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
