#!/usr/bin/env python3
"""Tests for Step 7E v02 Keyframe contract validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from step7.export_step7e_geometric_occlusion_evidence_v02 import (
    validate_keyframe_contract,
)


def write_inputs(
    directory: Path,
    keyframe_text: str,
    keyframe_count: int,
    digest: str | None = None,
    producer_step: int = 5,
    contract_version: str = "0.1",
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
                "contract_version": contract_version,
                "producer_step": producer_step,
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


def example_keyframes() -> list[dict[str, object]]:
    return [
        {
            "anchor_id": "a",
            "clip_id": "clip",
            "anchor_ns": 1,
        },
        {
            "anchor_id": "b",
            "clip_id": "clip",
            "anchor_ns": 2,
        },
    ]


def test_valid_keyframe_contract_is_accepted(
    tmp_path: Path,
):
    text = (
        '{"anchor_id":"a"}\n'
        '{"anchor_id":"b"}\n'
    )
    contract_path, keyframe_path = write_inputs(
        tmp_path,
        text,
        2,
    )

    contract = validate_keyframe_contract(
        contract_path=contract_path,
        keyframe_path=keyframe_path,
        keyframes=example_keyframes(),
    )

    assert contract["keyframe_count"] == 2


def test_wrong_keyframe_digest_is_rejected(
    tmp_path: Path,
):
    contract_path, keyframe_path = write_inputs(
        tmp_path,
        '{"anchor_id":"a"}\n',
        1,
        digest="0".ljust(64, "0"),
    )

    with pytest.raises(
        ValueError,
        match="SHA-256 mismatch",
    ):
        validate_keyframe_contract(
            contract_path=contract_path,
            keyframe_path=keyframe_path,
            keyframes=[
                {
                    "anchor_id": "a",
                    "clip_id": "clip",
                    "anchor_ns": 1,
                }
            ],
        )


def test_wrong_keyframe_count_is_rejected(
    tmp_path: Path,
):
    text = (
        '{"anchor_id":"a"}\n'
        '{"anchor_id":"b"}\n'
    )
    contract_path, keyframe_path = write_inputs(
        tmp_path,
        text,
        3,
    )

    with pytest.raises(
        ValueError,
        match="count mismatch",
    ):
        validate_keyframe_contract(
            contract_path=contract_path,
            keyframe_path=keyframe_path,
            keyframes=example_keyframes(),
        )


def test_duplicate_anchor_id_is_rejected(
    tmp_path: Path,
):
    text = (
        '{"anchor_id":"a"}\n'
        '{"anchor_id":"a"}\n'
    )
    contract_path, keyframe_path = write_inputs(
        tmp_path,
        text,
        2,
    )

    duplicate_keyframes = [
        {
            "anchor_id": "a",
            "clip_id": "clip",
            "anchor_ns": 1,
        },
        {
            "anchor_id": "a",
            "clip_id": "clip",
            "anchor_ns": 2,
        },
    ]

    with pytest.raises(
        ValueError,
        match="duplicate anchor_id",
    ):
        validate_keyframe_contract(
            contract_path=contract_path,
            keyframe_path=keyframe_path,
            keyframes=duplicate_keyframes,
        )


def test_wrong_contract_version_is_rejected(
    tmp_path: Path,
):
    text = (
        '{"anchor_id":"a"}\n'
        '{"anchor_id":"b"}\n'
    )
    contract_path, keyframe_path = write_inputs(
        tmp_path,
        text,
        2,
        contract_version="9.9",
    )

    with pytest.raises(
        ValueError,
        match="unsupported",
    ):
        validate_keyframe_contract(
            contract_path=contract_path,
            keyframe_path=keyframe_path,
            keyframes=example_keyframes(),
        )


def test_wrong_producer_step_is_rejected(
    tmp_path: Path,
):
    text = (
        '{"anchor_id":"a"}\n'
        '{"anchor_id":"b"}\n'
    )
    contract_path, keyframe_path = write_inputs(
        tmp_path,
        text,
        2,
        producer_step=4,
    )

    with pytest.raises(
        ValueError,
        match="producer_step",
    ):
        validate_keyframe_contract(
            contract_path=contract_path,
            keyframe_path=keyframe_path,
            keyframes=example_keyframes(),
        )
