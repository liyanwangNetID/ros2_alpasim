#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_meta_actions_v02 import step4_feature_commands


class FullStep4EntryTests(unittest.TestCase):
    def test_frozen_feature_sequence(self):
        commands = step4_feature_commands()
        scripts = [Path(command[1]).name for command in commands]
        self.assertEqual(
            scripts,
            [
                "profile_lane_matching_features.py",
                "refine_lane_matching_features.py",
                "profile_lateral_action_features.py",
                "profile_meta_action_features.py",
                "profile_lane_change_geometry_features.py",
            ],
        )
        self.assertEqual(commands[-1][-2:], ("--all", "--force"))

    def test_legacy_label_chain_is_not_called(self):
        rendered = "\n".join(" ".join(command) for command in step4_feature_commands())
        for legacy in (
            "generate_meta_actions.py",
            "evaluate_lateral_shadow_rules.py",
            "finalize_meta_actions_v02.py",
        ):
            self.assertNotIn(legacy, rendered)


if __name__ == "__main__":
    unittest.main(verbosity=2)
