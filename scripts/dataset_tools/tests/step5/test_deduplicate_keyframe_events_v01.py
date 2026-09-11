#!/usr/bin/env python3
import unittest


from step5.deduplicate_keyframe_events_v01 import normalize_anchor_events


class DedupRuleTests(unittest.TestCase):
    def test_exact_semantic_duplicates_merge(self):
        events = normalize_anchor_events([
            {"type": "turn_start", "direction": "left", "confidence": "high", "source": "a", "reasons": ["x"], "metrics": {}},
            {"type": "turn_start", "direction": "left", "confidence": "high", "source": "a", "reasons": ["x"], "metrics": {}},
        ])
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["merged_duplicate_count"], 2)

    def test_different_event_types_remain(self):
        events = normalize_anchor_events([
            {"type": "lateral_action_transition", "confidence": "high", "source": "a", "reasons": [], "metrics": {}},
            {"type": "lane_change_start", "direction": "left", "confidence": "high", "source": "a", "reasons": [], "metrics": {}},
        ])
        self.assertEqual(len(events), 2)

    def test_different_directions_remain(self):
        events = normalize_anchor_events([
            {"type": "lane_change_in_progress", "direction": "left", "confidence": "high", "source": "a", "reasons": [], "metrics": {}},
            {"type": "lane_change_in_progress", "direction": "right", "confidence": "high", "source": "a", "reasons": [], "metrics": {}},
        ])
        self.assertEqual(len(events), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
