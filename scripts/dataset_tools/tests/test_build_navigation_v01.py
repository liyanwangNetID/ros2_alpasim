#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_navigation_v01 import run_step6_pipeline, step6_commands


class Step6UnifiedEntryTests(unittest.TestCase):
    def test_pipeline_order(self):
        commands = step6_commands(force=False)
        self.assertEqual(
            [command[1] for command in commands],
            [
                "profile_navigation_branch_context_v01.py",
                "profile_road_level_navigation_features_v01.py",
                "profile_navigation_route_features_v01.py",
                "generate_navigation_candidates_v01.py",
                "finalize_navigation_v01.py",
            ],
        )

    def test_force_is_forwarded(self):
        commands = step6_commands(force=True)
        self.assertTrue(all(command[-1] == "--force" for command in commands))

    @patch("build_navigation_v01.subprocess.run")
    def test_pipeline_stops_on_failure(self, run_mock):
        run_mock.return_value.returncode = 9
        with self.assertRaisesRegex(RuntimeError, "exit code 9"):
            run_step6_pipeline(force=False)
        self.assertEqual(run_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
