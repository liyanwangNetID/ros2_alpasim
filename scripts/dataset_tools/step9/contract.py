from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from jsonschema import Draft202012Validator

REASONING_FORMAT_VERSION = "0.1-draft"
GENERATOR_VERSION = "0.1.0"
PROMPT_VERSION = "step9_local_reasoning_prompt_v0.2"

class Step9ContractError(ValueError):
    pass

def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def response_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object", "additionalProperties": False,
        "required": ["reasoning_summary", "used_evidence_keys", "limitations"],
        "properties": {
            "reasoning_summary": {"type": "string", "minLength": 1, "maxLength": 600},
            "used_evidence_keys": {"type": "array", "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{2,95}$"}, "uniqueItems": True},
            "limitations": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 240}, "uniqueItems": True},
        },
    }

def output_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema", "$id": "reasoning_schema_v0.1-draft.json",
        "title": "AlpaSim Step 9 Reasoning v0.1-draft", "type": "object", "additionalProperties": False,
        "required": ["reasoning_format_version", "generator_version", "prompt_version", "anchor_id", "clip_id", "anchor_ns", "reasoning_summary", "used_evidence_keys", "quality_status", "limitations", "source_record_sha256"],
        "properties": {
            "reasoning_format_version": {"const": REASONING_FORMAT_VERSION},
            "generator_version": {"const": GENERATOR_VERSION},
            "prompt_version": {"const": PROMPT_VERSION},
            "anchor_id": {"type": "string", "minLength": 1},
            "clip_id": {"type": "string", "pattern": "^test_clip_[0-9]+$"},
            "anchor_ns": {"type": "integer", "minimum": 0},
            "reasoning_summary": {"type": "string", "minLength": 1, "maxLength": 600},
            "used_evidence_keys": {"type": "array", "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{2,95}$"}, "uniqueItems": True},
            "quality_status": {"type": "string", "enum": ["usable", "partial", "unknown"]},
            "limitations": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 240}, "uniqueItems": True},
            "source_record_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    }

def read_input_records(path: Path, contract_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file(): raise Step9ContractError(f"missing Step 8 input: {path}")
    if not contract_path.is_file(): raise Step9ContractError(f"missing Step 8 contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    actual_hash = sha256_file(path)
    if contract.get("output_sha256") != actual_hash: raise Step9ContractError("Step 8 output SHA-256 mismatch")
    rows=[]; seen=set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip(): continue
            row=json.loads(line); anchor=row.get("anchor_id")
            if not isinstance(anchor, str) or not anchor: raise Step9ContractError(f"line {line_number}: invalid anchor_id")
            if anchor in seen: raise Step9ContractError(f"line {line_number}: duplicate anchor_id {anchor}")
            quality=(row.get("quality") or {}).get("status")
            if quality not in {"usable","partial","unknown"}: raise Step9ContractError(f"{anchor}: invalid quality status")
            seen.add(anchor); rows.append(row)
    if len(rows) != contract.get("record_count"): raise Step9ContractError("Step 8 record count mismatch")
    return rows, contract

def source_record_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()

def validate_schema(instance: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    errors=sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda e:list(e.absolute_path))
    if errors:
        error=errors[0]; location=".".join(str(x) for x in error.absolute_path) or "$"
        raise Step9ContractError(f"schema error at {location}: {error.message}")
