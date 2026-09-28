from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .checkpoint import load_checkpoint
from .config import Step9Config
from .contract import (
    output_schema,
    read_input_records,
    sha256_file,
    source_record_sha256,
)
from .prompt import build_evidence_package
from .validator import validate_response

FORBIDDEN_SUMMARY_TERMS = (
    "link_",
    "node_",
    "rule_",
    "track_id",
    "sha256",
    ".jsonl",
    "provenance",
)
LONGITUDINAL_PATTERNS = (
    r"\baccelerat(?:e|es|ed|ing|ion)\b",
    r"\bmaintain(?:s|ed|ing)? speed\b",
    r"\bdecelerat(?:e|es|ed|ing|ion)\b",
    r"\bslow(?:s|ed|ing)? down\b",
    r"\bstop(?:s|ped|ping)?\b",
    r"\blongitudinal\b",
)
UNCERTAINTY_TERMS = (
    "unknown",
    "uncertain",
    "insufficient",
    "limited",
    "limitation",
    "unavailable",
    "cannot determine",
    "cannot be determined",
    "unable to determine",
    "not enough evidence",
    "lack of evidence",
    "lacks evidence",
    "no supported causal link",
    "with caution",
    "cautious interpretation",
    "interpret cautiously",
    "interpreted cautiously",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def canonical_digest(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def has_longitudinal_support(source: dict[str, Any]) -> bool:
    nodes = {node["node_id"]: node for node in source["structured_coc"]["nodes"]}
    return any(
        link.get("relation") == "supports"
        and nodes[link["target_node_id"]].get("node_type") == "longitudinal_decision"
        for link in source["structured_coc"]["links"]
    )


def mentions_longitudinal(summary: str) -> bool:
    normalized = summary.lower().replace("_", " ")
    return any(re.search(pattern, normalized) for pattern in LONGITUDINAL_PATTERNS)


def select_review_rows(
    output_rows: list[dict[str, Any]],
    source_by_anchor: dict[str, dict[str, Any]],
    per_stratum: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, bool], list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        summary = row["reasoning_summary"]
        key = (
            row["quality_status"],
            "with_limitations" if row["limitations"] else "without_limitations",
            mentions_longitudinal(summary),
        )
        groups[key].append(row)

    selected = []
    for key in sorted(groups, key=str):
        for row in groups[key][:per_stratum]:
            source = source_by_anchor[row["anchor_id"]]
            selected.append(
                {
                    "anchor_id": row["anchor_id"],
                    "clip_id": row["clip_id"],
                    "quality_status": row["quality_status"],
                    "limitations": row["limitations"],
                    "reasoning_summary": row["reasoning_summary"],
                    "used_evidence_keys": row["used_evidence_keys"],
                    "has_longitudinal_support": has_longitudinal_support(source),
                    "review_stratum": {
                        "quality_status": key[0],
                        "limitation_group": key[1],
                        "mentions_longitudinal": key[2],
                    },
                }
            )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit frozen-candidate Step 9 production artifacts.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--review-per-stratum", type=int, default=4)
    args = parser.parse_args()
    if args.review_per_stratum < 1:
        raise SystemExit("--review-per-stratum must be positive")

    config = Step9Config.from_environment(args.data_root)
    sources, input_contract = read_input_records(config.input_path, config.input_contract_path)
    source_by_anchor = {row["anchor_id"]: row for row in sources}
    output_rows = read_jsonl(config.output_path)
    checkpoint = load_checkpoint(config.checkpoint_path)
    schema_file = json.loads(config.schema_path.read_text(encoding="utf-8"))
    summary_file = json.loads(config.summary_path.read_text(encoding="utf-8"))
    contract_file = json.loads(config.contract_path.read_text(encoding="utf-8"))

    errors: list[str] = []
    warnings: list[str] = []
    output_ids = [row.get("anchor_id") for row in output_rows]
    source_ids = [row["anchor_id"] for row in sources]

    if len(output_rows) != len(sources):
        errors.append(f"record count mismatch: output={len(output_rows)} input={len(sources)}")
    if len(set(output_ids)) != len(output_ids):
        errors.append("duplicate anchor_id in reasoning output")
    if output_ids != source_ids:
        errors.append("reasoning output anchor order differs from Step 8 input")
    if len(checkpoint) != len(sources):
        errors.append(f"checkpoint unique count mismatch: {len(checkpoint)} != {len(sources)}")

    validator = Draft202012Validator(output_schema())
    schema_errors = 0
    checkpoint_mismatches = 0
    source_hash_mismatches = 0
    forbidden_rows = []
    unsupported_longitudinal_rows = []
    uncertainty_rows_missing = []
    quality_counts = Counter()
    limitations_counts = Counter()
    evidence_key_counts = Counter()

    for index, row in enumerate(output_rows):
        row_errors = list(validator.iter_errors(row))
        schema_errors += len(row_errors)
        anchor = row.get("anchor_id")
        source = source_by_anchor.get(anchor)
        cached = checkpoint.get(anchor)
        if cached != row:
            checkpoint_mismatches += 1
        if source is None or row.get("source_record_sha256") != source_record_sha256(source):
            source_hash_mismatches += 1

        summary = str(row.get("reasoning_summary", ""))
        lowered = summary.lower()
        bad_terms = [term for term in FORBIDDEN_SUMMARY_TERMS if term in lowered]
        if bad_terms:
            forbidden_rows.append({"anchor_id": anchor, "terms": bad_terms})
        if source is not None:
            package = build_evidence_package(source)
            response = {
                "reasoning_summary": row["reasoning_summary"],
                "used_evidence_keys": row["used_evidence_keys"],
                "limitations": row["limitations"],
            }
            try:
                validate_response(response, package)
            except Exception as error:
                if str(error) == "summary claims support for a longitudinal action without longitudinal evidence":
                    unsupported_longitudinal_rows.append(anchor)
        quality = row.get("quality_status")
        limitations = row.get("limitations") or []
        if quality in {"partial", "unknown"} and not any(term in lowered for term in UNCERTAINTY_TERMS):
            uncertainty_rows_missing.append(anchor)
        quality_counts[quality] += 1
        limitations_counts["with_limitations" if limitations else "without_limitations"] += 1
        evidence_key_counts.update(row.get("used_evidence_keys") or [])

    if schema_errors:
        errors.append(f"output schema validation errors: {schema_errors}")
    if checkpoint_mismatches:
        errors.append(f"checkpoint/output mismatches: {checkpoint_mismatches}")
    if source_hash_mismatches:
        errors.append(f"source hash mismatches: {source_hash_mismatches}")
    if forbidden_rows:
        errors.append(f"rows with forbidden summary terms: {len(forbidden_rows)}")
    if unsupported_longitudinal_rows:
        errors.append(f"rows mentioning longitudinal action without support: {len(unsupported_longitudinal_rows)}")
    if uncertainty_rows_missing:
        errors.append(f"partial/unknown rows missing uncertainty language: {len(uncertainty_rows_missing)}")

    output_sha = sha256_file(config.output_path)
    schema_sha = sha256_file(config.schema_path)
    if summary_file.get("record_count") != len(output_rows):
        errors.append("summary record_count mismatch")
    if summary_file.get("output_sha256") != output_sha:
        errors.append("summary output_sha256 mismatch")
    if summary_file.get("schema_sha256") != schema_sha:
        errors.append("summary schema_sha256 mismatch")
    if summary_file.get("quality_status_counts") != dict(sorted(quality_counts.items())):
        errors.append("summary quality_status_counts mismatch")

    contract_checks = {
        "record_count": len(output_rows),
        "input_sha256": input_contract["output_sha256"],
        "output_sha256": output_sha,
        "schema_sha256": schema_sha,
    }
    for key, expected in contract_checks.items():
        if contract_file.get(key) != expected:
            errors.append(f"contract {key} mismatch")

    if schema_file != output_schema():
        errors.append("schema file differs from current output_schema()")

    review_rows = select_review_rows(output_rows, source_by_anchor, args.review_per_stratum)
    review_path = config.work_root / "production_review_sample_v01.jsonl"
    report_path = config.work_root / "production_audit_v01.json"
    review_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in review_rows),
        encoding="utf-8",
    )

    report = {
        "audit_version": "step9_production_audit_v0.1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "record_count": len(output_rows),
        "unique_anchor_count": len(set(output_ids)),
        "checkpoint_unique_count": len(checkpoint),
        "schema_error_count": schema_errors,
        "checkpoint_mismatch_count": checkpoint_mismatches,
        "source_hash_mismatch_count": source_hash_mismatches,
        "forbidden_summary_row_count": len(forbidden_rows),
        "unsupported_longitudinal_row_count": len(unsupported_longitudinal_rows),
        "uncertainty_missing_row_count": len(uncertainty_rows_missing),
        "quality_status_counts": dict(sorted(quality_counts.items())),
        "limitation_presence_counts": dict(sorted(limitations_counts.items())),
        "used_evidence_key_counts": dict(evidence_key_counts.most_common()),
        "output_path": str(config.output_path),
        "output_sha256": output_sha,
        "schema_path": str(config.schema_path),
        "schema_sha256": schema_sha,
        "summary_path": str(config.summary_path),
        "contract_path": str(config.contract_path),
        "review_sample_path": str(review_path),
        "review_sample_count": len(review_rows),
        "diagnostic_details": {
            "forbidden_rows": forbidden_rows[:20],
            "unsupported_longitudinal_anchors": unsupported_longitudinal_rows[:20],
            "uncertainty_missing_anchors": uncertainty_rows_missing[:20],
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("audit report:", report_path)
    print("review sample:", review_path)
    print("status:", report["status"])
    print("records:", report["record_count"])
    print("checkpoint:", report["checkpoint_unique_count"])
    print("review samples:", report["review_sample_count"])
    if errors:
        for error in errors:
            print("ERROR:", error)
        return 2
    print("PASS: Step 9 production artifacts, hashes, schema, checkpoint, and semantic guards are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
