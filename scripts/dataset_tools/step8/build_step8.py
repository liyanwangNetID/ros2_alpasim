#!/usr/bin/env python3
"""Build, validate, summarize, and contract Step 8 Structured CoC records."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator

from .causality import build_structured_causality
from .contract import (
    GENERATOR_VERSION,
    RULE_VERSION,
    STRUCTURED_CAUSALITY_FORMAT_VERSION,
    Step8Paths,
    load_and_validate_inputs,
)

OUTPUT_NAME = "structured_causality.jsonl"
SCHEMA_NAME = "structured_causality_schema_v0.1-draft.json"
SUMMARY_NAME = "step8_structured_causality_summary_v01.json"
CONTRACT_NAME = "structured_causality_contract_v0.1.json"
PROGRESS_INTERVAL_RECORDS = 250
PROGRESS_INTERVAL_SECONDS = 10.0


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duration_text(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def structured_causality_schema() -> dict[str, Any]:
    quality = {"type": "string", "enum": ["usable", "partial", "unknown"]}
    source_quality = {"type": "string", "enum": ["usable", "unknown"]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_NAME,
        "title": "AlpaSim Structured Causality v0.1-draft",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "structured_causality_format_version", "generator_version",
            "rule_version", "anchor_id", "clip_id", "anchor_ns",
            "structured_coc", "evidence_refs", "quality", "source_versions",
        ],
        "properties": {
            "structured_causality_format_version": {"const": STRUCTURED_CAUSALITY_FORMAT_VERSION},
            "generator_version": {"type": "string", "minLength": 1},
            "rule_version": {"type": "string", "minLength": 1},
            "anchor_id": {"type": "string", "minLength": 1},
            "clip_id": {"type": "string", "pattern": "^test_clip_[0-9]+$"},
            "anchor_ns": {"type": "integer", "minimum": 0},
            "structured_coc": {
                "type": "object", "additionalProperties": False,
                "required": ["nodes", "links"],
                "properties": {
                    "nodes": {"type": "array", "items": {"$ref": "#/$defs/node"}, "minItems": 3},
                    "links": {"type": "array", "items": {"$ref": "#/$defs/link"}, "minItems": 1},
                },
            },
            "evidence_refs": {
                "type": "object", "additionalProperties": False,
                "required": ["navigation_anchor_id", "scene_fact_anchor_id", "meta_action_anchor_id"],
                "properties": {
                    "navigation_anchor_id": {"type": "string", "minLength": 1},
                    "scene_fact_anchor_id": {"type": "string", "minLength": 1},
                    "meta_action_anchor_id": {"type": "string", "minLength": 1},
                },
            },
            "quality": {
                "type": "object", "additionalProperties": False,
                "required": ["status", "source_quality", "reasons"],
                "properties": {
                    "status": quality,
                    "source_quality": {
                        "type": "object", "additionalProperties": False,
                        "required": ["navigation", "scene_facts", "meta_action"],
                        "properties": {
                            "navigation": source_quality,
                            "scene_facts": source_quality,
                            "meta_action": source_quality,
                        },
                    },
                    "reasons": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": True},
                },
            },
            "source_versions": {
                "type": "object", "additionalProperties": False,
                "required": ["navigation_format_version", "scene_fact_format_version", "meta_action_format_version"],
                "properties": {
                    "navigation_format_version": {"type": ["string", "null"]},
                    "scene_fact_format_version": {"type": ["string", "null"]},
                    "meta_action_format_version": {"type": ["string", "null"]},
                },
            },
        },
        "$defs": {
            "node": {
                "type": "object", "additionalProperties": False,
                "required": ["node_id", "node_type", "value", "source_ref"],
                "properties": {
                    "node_id": {"type": "string", "minLength": 1},
                    "node_type": {"type": "string", "enum": ["navigation_intent", "longitudinal_decision", "lateral_decision", "actor_state"]},
                    "value": {"type": "object"},
                    "source_ref": {"type": "string", "minLength": 1},
                },
            },
            "link": {
                "type": "object", "additionalProperties": False,
                "required": ["link_id", "source_node_id", "target_node_id", "relation", "confidence", "rule_id", "reasons"],
                "properties": {
                    "link_id": {"type": "string", "pattern": "^link_[0-9]{3}$"},
                    "source_node_id": {"type": "string", "minLength": 1},
                    "target_node_id": {"type": "string", "minLength": 1},
                    "relation": {"type": "string", "enum": ["aligns_with", "supports", "insufficient_evidence"]},
                    "confidence": {"type": "string", "enum": ["supported", "weak", "unknown"]},
                    "rule_id": {"type": "string", "minLength": 1},
                    "reasons": {"type": "array", "items": {"type": "string", "minLength": 1}, "minItems": 1, "uniqueItems": True},
                },
            },
        },
    }


def validate_record_invariants(record: Mapping[str, Any]) -> None:
    anchor_id = record["anchor_id"]
    refs = record["evidence_refs"]
    if any(refs[name] != anchor_id for name in refs):
        raise ValueError(f"evidence reference mismatch for {anchor_id}")
    nodes = record["structured_coc"]["nodes"]
    links = record["structured_coc"]["links"]
    node_ids = [node["node_id"] for node in nodes]
    link_ids = [link["link_id"] for link in links]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"duplicate node_id for {anchor_id}")
    if len(link_ids) != len(set(link_ids)):
        raise ValueError(f"duplicate link_id for {anchor_id}")
    available = set(node_ids)
    for link in links:
        if link["source_node_id"] not in available or link["target_node_id"] not in available:
            raise ValueError(f"dangling link for {anchor_id}: {link['link_id']}")
    used = {link["source_node_id"] for link in links} | {link["target_node_id"] for link in links}
    actor_nodes = {node["node_id"] for node in nodes if node["node_type"] == "actor_state"}
    if actor_nodes - used:
        raise ValueError(f"unlinked Actor nodes for {anchor_id}: {sorted(actor_nodes - used)}")


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    quality = Counter(row["quality"]["status"] for row in rows)
    relations = Counter()
    confidence = Counter()
    rules = Counter()
    node_types = Counter()
    link_counts = Counter()
    actor_counts = Counter()
    for row in rows:
        nodes = row["structured_coc"]["nodes"]
        links = row["structured_coc"]["links"]
        link_counts[str(len(links))] += 1
        actor_counts[str(sum(node["node_type"] == "actor_state" for node in nodes))] += 1
        for node in nodes:
            node_types[node["node_type"]] += 1
        for link in links:
            relations[link["relation"]] += 1
            confidence[link["confidence"]] += 1
            rules[link["rule_id"]] += 1
    return {
        "summary_format_version": "0.1",
        "record_count": len(rows),
        "quality_status_counts": dict(sorted(quality.items())),
        "relation_counts": dict(sorted(relations.items())),
        "confidence_counts": dict(sorted(confidence.items())),
        "rule_counts": dict(sorted(rules.items())),
        "node_type_counts": dict(sorted(node_types.items())),
        "link_count_distribution": dict(sorted(link_counts.items(), key=lambda item: int(item[0]))),
        "actor_node_count_distribution": dict(sorted(actor_counts.items(), key=lambda item: int(item[0]))),
    }


def output_paths(paths: Step8Paths) -> dict[str, Path]:
    return {
        "output": paths.data_root / "annotations" / "v0.1-draft" / OUTPUT_NAME,
        "schema": paths.data_root / "schemas" / SCHEMA_NAME,
        "summary": paths.data_root / "reports" / SUMMARY_NAME,
        "contract": paths.data_root / "manifests" / CONTRACT_NAME,
    }


def atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    return temporary


def build(paths: Step8Paths, *, force: bool) -> dict[str, Any]:
    destinations = output_paths(paths)
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not force:
        raise FileExistsError("Step 8 outputs already exist; use --force: " + ", ".join(str(path) for path in existing))

    joined_rows, provenance = load_and_validate_inputs(paths)
    schema = structured_causality_schema()
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    records = []
    started = time.monotonic()
    last_progress = started
    total = len(joined_rows)
    for index, joined in enumerate(joined_rows, 1):
        record = build_structured_causality(joined)
        validate_record_invariants(record)
        errors = sorted(validator.iter_errors(record), key=lambda error: list(error.absolute_path))
        if errors:
            error = errors[0]
            location = ".".join(str(item) for item in error.absolute_path) or "$"
            raise ValueError(f"{joined.anchor_id} schema error at {location}: {error.message}")
        records.append(record)
        now = time.monotonic()
        if index == total or index % PROGRESS_INTERVAL_RECORDS == 0 or now - last_progress >= PROGRESS_INTERVAL_SECONDS:
            elapsed = now - started
            rate = index / elapsed if elapsed else 0.0
            eta = (total - index) / rate if rate else 0.0
            print(f"[Step 8] {index}/{total} | {100.0 * index / total:5.1f}% | elapsed {duration_text(elapsed)} | rate {rate:.1f}/s | ETA {duration_text(eta)}", flush=True)
            last_progress = now

    output_text = "".join(canonical_json(row) + "\n" for row in records)
    schema_text = json.dumps(schema, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    summary = summarize(records)
    summary.update({
        "output_path": str(destinations["output"]),
        "schema_path": str(destinations["schema"]),
        "source_sha256": provenance["source_sha256"],
        "elapsed_seconds": time.monotonic() - started,
    })

    temporary = {}
    try:
        temporary["output"] = atomic_write_text(destinations["output"], output_text)
        temporary["schema"] = atomic_write_text(destinations["schema"], schema_text)
        output_sha = sha256_file(temporary["output"])
        schema_sha = sha256_file(temporary["schema"])
        summary.update({"output_sha256": output_sha, "schema_sha256": schema_sha})
        temporary["summary"] = atomic_write_text(destinations["summary"], json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        contract = {
            "contract_version": "0.1",
            "producer_step": 8,
            "record_count": len(records),
            "structured_causality_format_version": STRUCTURED_CAUSALITY_FORMAT_VERSION,
            "generator_version": GENERATOR_VERSION,
            "rule_version": RULE_VERSION,
            "output_path": str(destinations["output"]),
            "output_sha256": output_sha,
            "schema_path": str(destinations["schema"]),
            "schema_sha256": schema_sha,
            "summary_path": str(destinations["summary"]),
            "source_sha256": provenance["source_sha256"],
            "keyframe_contract": provenance["keyframe_contract"],
            "future_actor_data_used": False,
            "future_ego_data_used": True,
            "meta_action_used_as_supervision": True,
        }
        temporary["contract"] = atomic_write_text(destinations["contract"], json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        for name in ("output", "schema", "summary", "contract"):
            os.replace(temporary[name], destinations[name])
    except BaseException:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise
    return {"destinations": destinations, "summary": summary, "contract": contract}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--data-root", type=Path, default=None)
    value.add_argument("--force", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    paths = Step8Paths.from_data_root(args.data_root)
    print("[Step 8] Validating frozen inputs", flush=True)
    result = build(paths, force=args.force)
    summary = result["summary"]
    print("Step 8 Structured CoC export")
    print("records:", summary["record_count"])
    print("quality:", summary["quality_status_counts"])
    print("relations:", summary["relation_counts"])
    print("rules:", summary["rule_counts"])
    print("sha256:", summary["output_sha256"])
    for name, path in result["destinations"].items():
        print(f"{name}:", path)
    print("PASS: Step 8 built, schema-validated, summarized, and contracted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
