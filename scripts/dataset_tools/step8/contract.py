"""Step 8 input contract and exact Anchor join.

This module is intentionally the only Step 8 component that knows the
upstream artifact layout. It validates frozen inputs before causality rules
are allowed to consume them.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

STRUCTURED_CAUSALITY_FORMAT_VERSION = "0.1-draft"
GENERATOR_VERSION = "0.1.0"
RULE_VERSION = "structured_causality_rules_v0.2"

EXPECTED_SHA256 = {
    "keyframes": "bb3cc755d537c0b8fa0c68aff457106ee00d583bff448bf737c2d286e0ccabf7",
    "meta_actions": "a07aacf417829e11d2fe437f01318d509d2d5a007a196440f3d9be95110f5973",
    "navigation": "bf7d91b283ef3ef93f421d24cdb267d2cbcfa7c241ac20f56ba7dad5bcaf572d",
    "scene_facts": "7b50f955058bbb159075b26a925bf5efc7aef15df189e11b649de63d8aa59fd8",
}


class Step8ContractError(ValueError):
    """Raised when an upstream artifact violates the Step 8 contract."""


@dataclass(frozen=True)
class Step8Paths:
    data_root: Path
    keyframes: Path
    meta_actions: Path
    navigation: Path
    scene_facts: Path
    keyframe_contract: Path

    @classmethod
    def from_data_root(cls, data_root: Path | str | None = None) -> "Step8Paths":
        root = Path(
            data_root
            or os.environ.get("ALPASIM_DATA_ROOT", "/home/lab/data_from_alpasim")
        ).expanduser().resolve()
        annotations = root / "annotations" / "v0.1-draft"
        return cls(
            data_root=root,
            keyframes=annotations / "keyframes.jsonl",
            meta_actions=annotations / "meta_actions_v0.2.jsonl",
            navigation=annotations / "navigation.jsonl",
            scene_facts=annotations / "scene_facts.jsonl",
            keyframe_contract=root / "manifests" / "keyframe_contract_v0.1.json",
        )

    def inputs(self) -> dict[str, Path]:
        return {
            "keyframes": self.keyframes,
            "meta_actions": self.meta_actions,
            "navigation": self.navigation,
            "scene_facts": self.scene_facts,
        }


@dataclass(frozen=True)
class JoinedAnchor:
    anchor_id: str
    clip_id: str
    anchor_ns: int
    keyframe: Mapping[str, Any]
    navigation: Mapping[str, Any]
    scene_fact: Mapping[str, Any]
    meta_action: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path, source_name: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    if not path.is_file():
        raise Step8ContractError(f"missing {source_name} artifact: {path}")
    order: list[str] = []
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise Step8ContractError(
                    f"{source_name}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise Step8ContractError(
                    f"{source_name}:{line_number}: row must be an object"
                )
            anchor_id = row.get("anchor_id")
            if not isinstance(anchor_id, str) or not anchor_id:
                raise Step8ContractError(
                    f"{source_name}:{line_number}: invalid anchor_id"
                )
            if anchor_id in rows:
                raise Step8ContractError(
                    f"{source_name}:{line_number}: duplicate anchor_id {anchor_id}"
                )
            order.append(anchor_id)
            rows[anchor_id] = row
    return order, rows


def _require_exact_coverage(
    source_name: str,
    keyframe_ids: set[str],
    source_ids: set[str],
) -> None:
    missing = sorted(keyframe_ids - source_ids)
    extra = sorted(source_ids - keyframe_ids)
    if missing or extra:
        raise Step8ContractError(
            f"{source_name} Anchor coverage mismatch: "
            f"missing={len(missing)} extra={len(extra)} "
            f"missing_examples={missing[:5]} extra_examples={extra[:5]}"
        )


def _require_meta_coverage(keyframe_ids: set[str], meta_ids: set[str]) -> None:
    missing = sorted(keyframe_ids - meta_ids)
    if missing:
        raise Step8ContractError(
            "meta_actions do not cover all Keyframes: "
            f"missing={len(missing)} examples={missing[:5]}"
        )


def _identity(row: Mapping[str, Any], source_name: str, anchor_id: str) -> tuple[str, int]:
    clip_id = row.get("clip_id")
    anchor_ns = row.get("anchor_ns")
    if not isinstance(clip_id, str) or not clip_id:
        raise Step8ContractError(f"{source_name}:{anchor_id}: invalid clip_id")
    if not isinstance(anchor_ns, int) or isinstance(anchor_ns, bool):
        raise Step8ContractError(f"{source_name}:{anchor_id}: invalid anchor_ns")
    return clip_id, anchor_ns


def validate_frozen_hashes(paths: Step8Paths) -> dict[str, str]:
    actual: dict[str, str] = {}
    for name, path in paths.inputs().items():
        if not path.is_file():
            raise Step8ContractError(f"missing {name} artifact: {path}")
        actual[name] = sha256_file(path)
        expected = EXPECTED_SHA256[name]
        if actual[name] != expected:
            raise Step8ContractError(
                f"{name} SHA-256 mismatch: expected={expected} actual={actual[name]}"
            )
    return actual


def validate_keyframe_contract(paths: Step8Paths, keyframe_count: int) -> dict[str, Any]:
    if not paths.keyframe_contract.is_file():
        raise Step8ContractError(
            f"missing Keyframe contract: {paths.keyframe_contract}"
        )
    with paths.keyframe_contract.open("r", encoding="utf-8") as handle:
        contract = json.load(handle)
    if contract.get("keyframe_count") != keyframe_count:
        raise Step8ContractError(
            "Keyframe contract count mismatch: "
            f"contract={contract.get('keyframe_count')} actual={keyframe_count}"
        )
    actual_keyframe_hash = sha256_file(paths.keyframes)
    if contract.get("keyframe_sha256") != actual_keyframe_hash:
        raise Step8ContractError("Keyframe contract SHA-256 mismatch")
    actual_meta_hash = sha256_file(paths.meta_actions)
    if contract.get("meta_action_sha256") != actual_meta_hash:
        raise Step8ContractError("Meta-action contract SHA-256 mismatch")
    return contract


def join_rows(
    keyframe_order: Iterable[str],
    keyframes: Mapping[str, Mapping[str, Any]],
    navigation: Mapping[str, Mapping[str, Any]],
    scene_facts: Mapping[str, Mapping[str, Any]],
    meta_actions: Mapping[str, Mapping[str, Any]],
) -> list[JoinedAnchor]:
    order = list(keyframe_order)
    keyframe_ids = set(order)
    if len(order) != len(keyframe_ids):
        raise Step8ContractError("keyframe_order contains duplicate Anchor IDs")
    if keyframe_ids != set(keyframes):
        raise Step8ContractError("keyframe_order does not match Keyframe rows")
    _require_exact_coverage("navigation", keyframe_ids, set(navigation))
    _require_exact_coverage("scene_facts", keyframe_ids, set(scene_facts))
    _require_meta_coverage(keyframe_ids, set(meta_actions))

    joined: list[JoinedAnchor] = []
    for anchor_id in order:
        sources = {
            "keyframes": keyframes[anchor_id],
            "navigation": navigation[anchor_id],
            "scene_facts": scene_facts[anchor_id],
            "meta_actions": meta_actions[anchor_id],
        }
        identities = {
            name: _identity(row, name, anchor_id) for name, row in sources.items()
        }
        expected = identities["keyframes"]
        inconsistent = {
            name: value for name, value in identities.items() if value != expected
        }
        if inconsistent:
            raise Step8ContractError(
                f"identity mismatch for {anchor_id}: "
                f"expected={expected} inconsistent={inconsistent}"
            )
        joined.append(
            JoinedAnchor(
                anchor_id=anchor_id,
                clip_id=expected[0],
                anchor_ns=expected[1],
                keyframe=sources["keyframes"],
                navigation=sources["navigation"],
                scene_fact=sources["scene_facts"],
                meta_action=sources["meta_actions"],
            )
        )
    return joined


def load_and_validate_inputs(
    paths: Step8Paths,
    *,
    validate_hashes: bool = True,
) -> tuple[list[JoinedAnchor], dict[str, Any]]:
    loaded: dict[str, tuple[list[str], dict[str, dict[str, Any]]]] = {
        name: read_jsonl(path, name) for name, path in paths.inputs().items()
    }
    keyframe_order, keyframes = loaded["keyframes"]
    joined = join_rows(
        keyframe_order,
        keyframes,
        loaded["navigation"][1],
        loaded["scene_facts"][1],
        loaded["meta_actions"][1],
    )
    contract = validate_keyframe_contract(paths, len(joined))
    hashes = validate_frozen_hashes(paths) if validate_hashes else {
        name: sha256_file(path) for name, path in paths.inputs().items()
    }
    provenance = {
        "keyframe_contract": contract,
        "source_sha256": hashes,
        "keyframe_count": len(joined),
    }
    return joined, provenance
