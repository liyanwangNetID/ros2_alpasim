#!/usr/bin/env python3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_keyframes_v01 import run_step5_pipeline, step5_commands


class Step5UnifiedEntryTests(unittest.TestCase):
    def test_full_pipeline_order(self):
        commands = step5_commands(force=False, reuse_existing_events=False)
        self.assertEqual(
            [command[1] for command in commands],
            [
                "detect_keyframe_events_v01.py",
                "deduplicate_keyframe_events_v01.py",
                "select_keyframes_v01.py",
            ],
        )

    def test_force_is_forwarded_to_all_stages(self):
        commands = step5_commands(force=True, reuse_existing_events=False)
        self.assertTrue(all(command[-1] == "--force" for command in commands))

    def test_reuse_existing_events_only_runs_selection(self):
        commands = step5_commands(force=False, reuse_existing_events=True)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][1], "select_keyframes_v01.py")

    @patch("build_keyframes_v01.subprocess.run")
    def test_pipeline_stops_on_stage_failure(self, run_mock):
        run_mock.return_value.returncode = 7
        with self.assertRaisesRegex(RuntimeError, "exit code 7"):
            run_step5_pipeline(force=False, reuse_existing_events=False)
        self.assertEqual(run_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
