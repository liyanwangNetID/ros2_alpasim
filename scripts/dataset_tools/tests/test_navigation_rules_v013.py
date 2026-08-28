#!/usr/bin/env python3
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from navigation_rules_v01 import classify_navigation


def route(signed_deg=30.0, lookahead=80.0):
    return {
        'quality_status': 'usable',
        'route_lookahead_distance_m': lookahead,
        'route': {
            'route_signed_heading_change_rad': math.radians(signed_deg),
        },
    }


def branch(distance, speed=0.937281608581543):
    return {
        'quality_status': 'usable',
        'anchor_speed_mps': speed,
        'branch_context': {
            'first_intersection_evidence': {
                'route_distance_m': 4.210254988037607,
                'evidence': ['branching'],
            },
            'first_observed_branch': {
                'route_distance_m': distance,
                'route_relation_to_natural': 'left_of_natural',
                'reliability_status': 'reliable',
            },
        },
    }


class NavigationRuleV013Tests(unittest.TestCase):
    def test_known_early_left_prompt_is_deferred(self):
        result = classify_navigation(
            route(signed_deg=69.07331806644386),
            branch(distance=16.84160502352037),
        )
        self.assertEqual(result['action'], 'straight')
        self.assertEqual(result['text'], 'Continue along the road.')
        self.assertEqual(
            result['decision_source'],
            'first_observed_branch_beyond_dynamic_preview_horizon',
        )

    def test_same_left_branch_inside_horizon_is_emitted(self):
        result = classify_navigation(
            route(signed_deg=69.0),
            branch(distance=14.0),
        )
        self.assertEqual(result['action'], 'left')
        self.assertEqual(
            result['text'],
            'Turn left at the upcoming intersection.',
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
