#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profile_lane_change_geometry_features import merge_feature_records


class DualInputTests(unittest.TestCase):
    def records(self):
        identity = {
            "anchor_id": "clip_1_100",
            "clip_id": "clip_1",
            "anchor_ns": 100,
            "future_horizon_ns": 300,
        }
        meta = dict(identity, feature_format_version="meta-v2")
        meta["lane_matching"] = {"transitions": [{"relation": "right_adjacent"}]}
        lateral = dict(identity, feature_format_version="lat-v3", map_id="map")
        lateral["lateral"] = {
            "topology": {"junction_evidence_level": "A"},
            "adjacent_transition_evidence": [],
        }
        return meta, lateral

    def test_merge_preserves_both_sources(self):
        meta, lateral = self.records()
        merged = merge_feature_records(meta, lateral)
        self.assertEqual(merged["lane_matching"]["transitions"][0]["relation"], "right_adjacent")
        self.assertEqual(merged["lateral"]["topology"]["junction_evidence_level"], "A")
        self.assertEqual(merged["source_feature_versions"]["meta"], "meta-v2")
        self.assertEqual(merged["source_feature_versions"]["lateral"], "lat-v3")

    def test_identity_mismatch_is_rejected(self):
        meta, lateral = self.records()
        lateral["anchor_ns"] = 101
        with self.assertRaises(ValueError):
            merge_feature_records(meta, lateral)

    def test_missing_required_sections_are_rejected(self):
        meta, lateral = self.records()
        del meta["lane_matching"]
        with self.assertRaises(ValueError):
            merge_feature_records(meta, lateral)


if __name__ == "__main__":
    unittest.main(verbosity=2)
