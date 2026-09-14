#!/usr/bin/env python3
"""Tests for the unified Step 6 production entry."""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

from step6.build_navigation_v01 import (
    run_step6_pipeline,
    step6_commands,
)


class Step6UnifiedEntryTests(unittest.TestCase):
    def test_pipeline_order(self):
        commands = step6_commands(force=False)

        self.assertEqual(
            [command[2] for command in commands],
            [
                "step6.profile_navigation_branch_context_v01",
                "step6.profile_road_level_navigation_features_v01",
                "step6.profile_navigation_route_features_v01",
                "step6.generate_navigation_v01",
            ],
        )

        for command in commands:
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[1], "-m")

    def test_force_is_forwarded(self):
        commands = step6_commands(force=True)

        self.assertTrue(
            all(
                command[-1] == "--force"
                for command in commands
            )
        )

    @patch("step6.build_navigation_v01.subprocess.run")
    def test_pipeline_stops_on_failure(
        self,
        run_mock,
    ):
        run_mock.return_value.returncode = 9

        with self.assertRaisesRegex(
            RuntimeError,
            "exit code 9",
        ):
            run_step6_pipeline(force=False)

        self.assertEqual(run_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
