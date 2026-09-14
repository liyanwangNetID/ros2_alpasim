#!/usr/bin/env python3
"""Tests for Step 6 Keyframe contract validation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from step6.generate_navigation_v01 import (
    validate_keyframe_contract,
)


def write_inputs(
    directory: Path,
    keyframe_text: str,
    keyframe_count: int,
    digest: str | None = None,
) -> tuple[Path, Path]:
    keyframe_path = directory / "keyframes.jsonl"
    contract_path = directory / "contract.json"

    keyframe_path.write_text(
        keyframe_text,
        encoding="utf-8",
    )

    actual_digest = hashlib.sha256(
        keyframe_text.encode("utf-8")
    ).hexdigest()

    contract_path.write_text(
        json.dumps(
            {
                "contract_version": "0.1",
                "producer_step": 5,
                "keyframe_count": keyframe_count,
                "keyframe_sha256": (
                    digest
                    if digest is not None
                    else actual_digest
                ),
            }
        ),
        encoding="utf-8",
    )

    return contract_path, keyframe_path


def test_valid_keyframe_contract_is_accepted():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        text = (
            '{"anchor_id":"a"}\n'
            '{"anchor_id":"b"}\n'
        )

        contract_path, keyframe_path = write_inputs(
            root,
            text,
            2,
        )

        contract = validate_keyframe_contract(
            contract_path,
            keyframe_path,
            {"a", "b"},
        )

        assert contract["keyframe_count"] == 2


def test_wrong_keyframe_digest_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        contract_path, keyframe_path = write_inputs(
            root,
            '{"anchor_id":"a"}\n',
            1,
            digest="0" * 64,
        )

        with pytest.raises(ValueError):
            validate_keyframe_contract(
                contract_path,
                keyframe_path,
                {"a"},
            )


def test_wrong_keyframe_count_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        contract_path, keyframe_path = write_inputs(
            root,
            '{"anchor_id":"a"}\n',
            2,
        )

        with pytest.raises(ValueError):
            validate_keyframe_contract(
                contract_path,
                keyframe_path,
                {"a"},
            )


def test_wrong_contract_producer_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        text = '{"anchor_id":"a"}\n'

        contract_path, keyframe_path = write_inputs(
            root,
            text,
            1,
        )

        value = json.loads(
            contract_path.read_text(encoding="utf-8")
        )
        value["producer_step"] = 4
        contract_path.write_text(
            json.dumps(value),
            encoding="utf-8",
        )

        with pytest.raises(ValueError):
            validate_keyframe_contract(
                contract_path,
                keyframe_path,
                {"a"},
            )
