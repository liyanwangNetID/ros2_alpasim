#!/usr/bin/env python3
"""Tests for Step 7D.5 projected convex-hull geometry."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


from step7.projected_hull_v01 import (  # noqa: E402
    Point2D,
    clip_polygon_to_image,
    convex_hull,
    polygon_area,
    summarize_projected_hull,
)


class ConvexHullTests(unittest.TestCase):
    def test_convex_hull_removes_internal_points(self) -> None:
        result = convex_hull(
            [
                Point2D(0.0, 0.0),
                Point2D(2.0, 0.0),
                Point2D(2.0, 2.0),
                Point2D(0.0, 2.0),
                Point2D(1.0, 1.0),
            ]
        )
        self.assertEqual(len(result), 4)
        self.assertEqual(set(result), {
            Point2D(0.0, 0.0), Point2D(2.0, 0.0),
            Point2D(2.0, 2.0), Point2D(0.0, 2.0),
        })

    def test_duplicate_points_are_removed(self) -> None:
        result = convex_hull(
            [Point2D(0.0, 0.0), Point2D(0.0, 0.0), Point2D(1.0, 0.0)]
        )
        self.assertEqual(result, (Point2D(0.0, 0.0), Point2D(1.0, 0.0)))


class PolygonAreaTests(unittest.TestCase):
    def test_rectangle_area(self) -> None:
        self.assertEqual(
            polygon_area(
                (
                    Point2D(0.0, 0.0), Point2D(4.0, 0.0),
                    Point2D(4.0, 3.0), Point2D(0.0, 3.0),
                )
            ),
            12.0,
        )

    def test_degenerate_polygon_area_is_zero(self) -> None:
        self.assertEqual(polygon_area((Point2D(0.0, 0.0),)), 0.0)


class ImageClipTests(unittest.TestCase):
    def test_polygon_fully_inside_is_unchanged_in_area(self) -> None:
        polygon = (
            Point2D(1.0, 1.0), Point2D(8.0, 1.0),
            Point2D(8.0, 8.0), Point2D(1.0, 8.0),
        )
        clipped = clip_polygon_to_image(polygon, width=10, height=10)
        self.assertAlmostEqual(polygon_area(clipped), polygon_area(polygon))

    def test_polygon_crossing_right_edge_is_clipped(self) -> None:
        polygon = (
            Point2D(5.0, 2.0), Point2D(15.0, 2.0),
            Point2D(15.0, 8.0), Point2D(5.0, 8.0),
        )
        clipped = clip_polygon_to_image(polygon, width=10, height=10)
        self.assertAlmostEqual(polygon_area(clipped), 24.0)
        self.assertTrue(all(point.u <= 9.0 for point in clipped))

    def test_polygon_fully_outside_becomes_empty(self) -> None:
        polygon = (
            Point2D(20.0, 2.0), Point2D(30.0, 2.0),
            Point2D(30.0, 8.0), Point2D(20.0, 8.0),
        )
        self.assertEqual(
            clip_polygon_to_image(polygon, width=10, height=10),
            (),
        )


class SummaryTests(unittest.TestCase):
    def test_inside_hull_ratio_is_one(self) -> None:
        result = summarize_projected_hull(
            [
                Point2D(1.0, 1.0), Point2D(8.0, 1.0),
                Point2D(8.0, 8.0), Point2D(1.0, 8.0),
            ],
            width=10,
            height=10,
        )
        self.assertAlmostEqual(result.inside_image_hull_ratio, 1.0)
        self.assertFalse(result.truncated_by_image)

    def test_partial_hull_has_fractional_ratio(self) -> None:
        result = summarize_projected_hull(
            [
                Point2D(5.0, 2.0), Point2D(15.0, 2.0),
                Point2D(15.0, 8.0), Point2D(5.0, 8.0),
            ],
            width=10,
            height=10,
        )
        self.assertAlmostEqual(result.projected_hull_area_px, 60.0)
        self.assertAlmostEqual(result.inside_image_hull_area_px, 24.0)
        self.assertAlmostEqual(result.inside_image_hull_ratio, 0.4)
        self.assertTrue(result.truncated_by_image)

    def test_to_dict_contains_point_objects_as_dicts(self) -> None:
        result = summarize_projected_hull(
            [Point2D(0.0, 0.0), Point2D(2.0, 0.0), Point2D(1.0, 2.0)],
            width=10,
            height=10,
        ).to_dict()
        self.assertIsInstance(result["projected_hull"], tuple)
        self.assertIsInstance(result["projected_hull"][0], dict)


if __name__ == "__main__":
    unittest.main(verbosity=2)
