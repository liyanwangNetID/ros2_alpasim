#!/usr/bin/env python3
"""Tests for Step 4 Candidate Anchor contract validation."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from project_paths import (
    ANNOTATION_ROOT,
    MANIFEST_ROOT,
)
from step4.build_meta_actions_v02 import (
    DEFAULT_ANCHOR_CONTRACT,
    DEFAULT_CANDIDATE_ANCHORS,
    parse_args,
    validate_candidate_anchor_contract,
)


class CandidateAnchorContractTests(unittest.TestCase):
    def write_inputs(
        self,
        directory: Path,
        anchor_text: str,
        *,
        count: int,
        digest: str | None = None,
    ) -> tuple[Path, Path]:
        anchor_path = directory / "anchors.jsonl"
        contract_path = directory / "contract.json"

        anchor_path.write_text(
            anchor_text,
            encoding="utf-8",
        )

        actual_digest = hashlib.sha256(
            anchor_text.encode("utf-8")
        ).hexdigest()

        contract = {
            "contract_version": "0.1",
            "producer_step": 3,
            "candidate_anchor_count": count,
            "candidate_anchor_sha256": (
                digest
                if digest is not None
                else actual_digest
            ),
        }

        contract_path.write_text(
            json.dumps(contract),
            encoding="utf-8",
        )

        return contract_path, anchor_path

    def test_valid_contract_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path, anchor_path = self.write_inputs(
                root,
                '{"anchor_id":"a"}\n'
                '{"anchor_id":"b"}\n',
                count=2,
            )

            contract, anchor_ids = (
                validate_candidate_anchor_contract(
                    contract_path=contract_path,
                    candidate_anchor_path=anchor_path,
                )
            )

            self.assertEqual(
                contract["candidate_anchor_count"],
                2,
            )
            self.assertEqual(anchor_ids, {"a", "b"})

    def test_wrong_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path, anchor_path = self.write_inputs(
                root,
                '{"anchor_id":"a"}\n',
                count=1,
                digest="0" * 64,
            )

            with self.assertRaises(ValueError):
                validate_candidate_anchor_contract(
                    contract_path=contract_path,
                    candidate_anchor_path=anchor_path,
                )

    def test_wrong_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract_path, anchor_path = self.write_inputs(
                root,
                '{"anchor_id":"a"}\n',
                count=2,
            )

            with self.assertRaises(ValueError):
                validate_candidate_anchor_contract(
                    contract_path=contract_path,
                    candidate_anchor_path=anchor_path,
                )

    def test_default_paths_use_project_configuration(self):
        with patch.object(
            sys,
            "argv",
            ["build_meta_actions_v02.py"],
        ):
            args = parse_args()

        self.assertEqual(
            args.candidate_anchor_input,
            ANNOTATION_ROOT / "candidate_anchors.jsonl",
        )
        self.assertEqual(
            args.anchor_contract,
            MANIFEST_ROOT
            / "candidate_anchor_contract_v0.1.json",
        )
        self.assertEqual(
            DEFAULT_CANDIDATE_ANCHORS,
            args.candidate_anchor_input,
        )
        self.assertEqual(
            DEFAULT_ANCHOR_CONTRACT,
            args.anchor_contract,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
