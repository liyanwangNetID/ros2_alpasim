#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from keyframe_selection_rules_v01 import select_with_clip_preference


class SelectionRuleTests(unittest.TestCase):
    def test_deterministic(self):
        records = [
            {"anchor_id": f"a{i}", "clip_id": f"c{i % 3}"}
            for i in range(10)
        ]
        first = select_with_clip_preference(records, quota=5, bucket="x")
        second = select_with_clip_preference(records, quota=5, bucket="x")
        self.assertEqual(first, second)

    def test_prefers_distinct_clips(self):
        records = [
            {"anchor_id": "a1", "clip_id": "c1"},
            {"anchor_id": "a2", "clip_id": "c1"},
            {"anchor_id": "a3", "clip_id": "c2"},
        ]
        chosen = select_with_clip_preference(records, quota=2, bucket="y")
        self.assertEqual(len({record["clip_id"] for record in chosen}), 2)

    def test_none_quota_keeps_all(self):
        records = [
            {"anchor_id": "a1", "clip_id": "c1"},
            {"anchor_id": "a2", "clip_id": "c2"},
        ]
        self.assertEqual(
            len(select_with_clip_preference(records, quota=None, bucket="z")),
            2,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
