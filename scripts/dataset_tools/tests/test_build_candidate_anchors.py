#!/usr/bin/env python3
"""Tests for Step 3 candidate-anchor path defaults."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from build_candidate_anchors import parse_args
from project_paths import (
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    MANIFEST_ROOT,
    REPORT_ROOT,
)


class BuildCandidateAnchorsPathTests(unittest.TestCase):
    def test_default_paths_come_from_project_configuration(self):
        with patch.object(sys, "argv", ["build_candidate_anchors.py"]):
            args = parse_args()

        self.assertEqual(args.dataset_root, ALPASIM_DATA_ROOT)
        self.assertEqual(args.manifest, MANIFEST_ROOT / "clips_v0.1.jsonl")
        self.assertEqual(
            args.anchor_output,
            ANNOTATION_ROOT / "candidate_anchors.jsonl",
        )
        self.assertEqual(
            args.per_clip_output,
            REPORT_ROOT / "candidate_anchor_per_clip_v0.1.jsonl",
        )
        self.assertEqual(
            args.summary_output,
            REPORT_ROOT / "candidate_anchor_summary_v0.1.json",
        )
        self.assertIsNone(args.limit_clips)
        self.assertFalse(args.force)

    def test_cli_paths_override_project_configuration(self):
        argv = [
            "build_candidate_anchors.py",
            "--dataset-root",
            "/portable/dataset",
            "--manifest",
            "/portable/input/clips.jsonl",
            "--anchor-output",
            "/portable/output/anchors.jsonl",
            "--per-clip-output",
            "/portable/output/per_clip.jsonl",
            "--summary-output",
            "/portable/output/summary.json",
            "--limit-clips",
            "3",
            "--force",
        ]

        with patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.dataset_root, Path("/portable/dataset"))
        self.assertEqual(args.manifest, Path("/portable/input/clips.jsonl"))
        self.assertEqual(
            args.anchor_output,
            Path("/portable/output/anchors.jsonl"),
        )
        self.assertEqual(
            args.per_clip_output,
            Path("/portable/output/per_clip.jsonl"),
        )
        self.assertEqual(
            args.summary_output,
            Path("/portable/output/summary.json"),
        )
        self.assertEqual(args.limit_clips, 3)
        self.assertTrue(args.force)


if __name__ == "__main__":
    unittest.main(verbosity=2)
