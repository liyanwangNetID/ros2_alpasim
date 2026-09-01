#!/usr/bin/env python3
"""Tests for the Step 1 clip-manifest command-line path defaults."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from build_clip_manifest import parse_args
from project_paths import ALPASIM_DATA_ROOT, MANIFEST_ROOT, REPORT_ROOT


class BuildClipManifestPathTests(unittest.TestCase):
    def test_default_paths_come_from_project_configuration(self):
        with patch.object(sys, "argv", ["build_clip_manifest.py"]):
            args = parse_args()

        self.assertEqual(args.dataset_root, ALPASIM_DATA_ROOT)
        self.assertEqual(
            args.manifest_output,
            MANIFEST_ROOT / "clips_v0.1.jsonl",
        )
        self.assertEqual(
            args.summary_output,
            REPORT_ROOT / "clip_manifest_summary_v0.1.json",
        )
        self.assertFalse(args.force)

    def test_cli_paths_override_project_configuration(self):
        dataset_root = Path("/portable/dataset")
        manifest_output = Path("/portable/output/clips.jsonl")
        summary_output = Path("/portable/output/summary.json")

        argv = [
            "build_clip_manifest.py",
            "--dataset-root",
            str(dataset_root),
            "--manifest-output",
            str(manifest_output),
            "--summary-output",
            str(summary_output),
            "--force",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.dataset_root, dataset_root)
        self.assertEqual(args.manifest_output, manifest_output)
        self.assertEqual(args.summary_output, summary_output)
        self.assertTrue(args.force)


if __name__ == "__main__":
    unittest.main(verbosity=2)
