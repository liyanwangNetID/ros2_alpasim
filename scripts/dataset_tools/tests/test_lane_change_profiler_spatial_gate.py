#!/usr/bin/env python3
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profile_lane_change_geometry_features import spatial_lane_fallback_enabled


class SpatialGateTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.pop("ENABLE_SPATIAL_LANE_FALLBACK", None)

    def tearDown(self):
        os.environ.pop("ENABLE_SPATIAL_LANE_FALLBACK", None)
        if self.previous is not None:
            os.environ["ENABLE_SPATIAL_LANE_FALLBACK"] = self.previous

    def test_disabled_by_default(self):
        self.assertFalse(spatial_lane_fallback_enabled())

    def test_enabled_explicitly(self):
        os.environ["ENABLE_SPATIAL_LANE_FALLBACK"] = "1"
        self.assertTrue(spatial_lane_fallback_enabled())

    def test_false_spellings(self):
        for value in ("0", "false", "no", "off"):
            os.environ["ENABLE_SPATIAL_LANE_FALLBACK"] = value
            self.assertFalse(spatial_lane_fallback_enabled())


if __name__ == "__main__":
    unittest.main(verbosity=2)
