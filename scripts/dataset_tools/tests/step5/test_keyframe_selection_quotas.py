#!/usr/bin/env python3
"""Tests for dynamic Step 5 Keyframe selection quotas."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import pytest

from step5.select_keyframes_v01 import (
    REFERENCE_ANCHOR_COUNT,
    scaled_quota,
    selection_quotas,
    validate_meta_action_contract,
)


def test_current_dataset_reproduces_existing_quotas():
    quotas = selection_quotas(
        REFERENCE_ANCHOR_COUNT
    )

    assert quotas["stable_lateral_quotas"] == {
        "turn_left": None,
        "turn_right": None,
        "change_lane_left": 50,
        "change_lane_right": 50,
    }
    assert quotas["stable_longitudinal_quotas"] == {
        "accelerate": 100,
        "decelerate": 100,
        "stop": 100,
    }
    assert quotas["normal_baseline_quota"] == 500


def test_double_dataset_doubles_finite_quotas():
    quotas = selection_quotas(
        REFERENCE_ANCHOR_COUNT
        + REFERENCE_ANCHOR_COUNT
    )

    assert (
        quotas["stable_lateral_quotas"][
            "change_lane_left"
        ]
        == 100
    )
    assert (
        quotas["stable_longitudinal_quotas"][
            "accelerate"
        ]
        == 200
    )
    assert quotas["normal_baseline_quota"] == 1000


def test_rare_turn_actions_remain_unlimited():
    quotas = selection_quotas(50_000)

    assert (
        quotas["stable_lateral_quotas"]["turn_left"]
        is None
    )
    assert (
        quotas["stable_lateral_quotas"]["turn_right"]
        is None
    )


def test_small_positive_dataset_has_positive_quotas():
    quotas = selection_quotas(1)

    assert (
        quotas["stable_lateral_quotas"][
            "change_lane_left"
        ]
        == 1
    )
    assert (
        quotas["stable_longitudinal_quotas"]["stop"]
        == 1
    )
    assert quotas["normal_baseline_quota"] == 1


def test_invalid_quota_inputs_are_rejected():
    with pytest.raises(ValueError):
        selection_quotas(0)

    with pytest.raises(ValueError):
        scaled_quota(100, 0)


def write_contract_inputs(
    directory: Path,
) -> tuple[Path, Path]:
    meta_text = (
        '{"anchor_id":"a"}\n'
        '{"anchor_id":"b"}\n'
    )
    meta_path = directory / "meta.jsonl"
    contract_path = directory / "contract.json"

    meta_path.write_text(
        meta_text,
        encoding="utf-8",
    )

    digest = hashlib.sha256(
        meta_text.encode("utf-8")
    ).hexdigest()

    contract_path.write_text(
        json.dumps(
            {
                "contract_version": "0.2",
                "producer_step": 4,
                "candidate_anchor_count": 2,
                "meta_action_count": 2,
                "meta_action_sha256": digest,
            }
        ),
        encoding="utf-8",
    )

    return contract_path, meta_path


def test_valid_meta_action_contract_is_accepted():
    with tempfile.TemporaryDirectory() as directory:
        contract_path, meta_path = write_contract_inputs(
            Path(directory)
        )

        contract = validate_meta_action_contract(
            contract_path,
            meta_path,
            2,
        )

        assert contract["meta_action_count"] == 2


def test_contract_count_mismatch_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        contract_path, meta_path = write_contract_inputs(
            Path(directory)
        )

        with pytest.raises(ValueError):
            validate_meta_action_contract(
                contract_path,
                meta_path,
                3,
            )
