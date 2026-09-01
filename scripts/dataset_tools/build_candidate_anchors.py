#!/usr/bin/env python3
"""Build candidate anchors for the AlpaSim driving dataset.

This is the executable tool for Step 3. It reads finalized clips from the clip
manifest, delegates per-clip anchor selection to AnchorSelector, and writes:

1. selected candidate anchors as JSONL;
2. per-clip selection statistics as JSONL;
3. an aggregate JSON summary.

The tool is read-only with respect to raw test_clip_NNN directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MODULE_DIRECTORY = Path(__file__).resolve().parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from anchor_selector import (  # noqa: E402
    AnchorSelectionResult,
    AnchorSelector,
    AnchorSelectorConfig,
)
from clip_reader import DrivingClipReader  # noqa: E402
from project_paths import (  # noqa: E402
    ALPASIM_DATA_ROOT,
    ANNOTATION_ROOT,
    MANIFEST_ROOT,
    REPORT_ROOT,
)


SCRIPT_VERSION = "0.1.0"
ANCHOR_FORMAT_VERSION = "0.1-draft"
DEFAULT_DATASET_ROOT = ALPASIM_DATA_ROOT
DEFAULT_MANIFEST = MANIFEST_ROOT / "clips_v0.1.jsonl"
DEFAULT_ANCHOR_OUTPUT = (
    ANNOTATION_ROOT / "candidate_anchors.jsonl"
)
DEFAULT_PER_CLIP_OUTPUT = (
    REPORT_ROOT / "candidate_anchor_per_clip_v0.1.jsonl"
)
DEFAULT_SUMMARY_OUTPUT = (
    REPORT_ROOT / "candidate_anchor_summary_v0.1.json"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL row is not an object at {path}:{line_number}"
                    )
                records.append(record)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"input manifest not found: {path}") from exc
    return records


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def distribution(values: Iterable[float]) -> dict[str, float | None]:
    items = list(values)
    return {
        "minimum": min(items) if items else None,
        "mean": statistics.mean(items) if items else None,
        "median": statistics.median(items) if items else None,
        "p90": percentile(items, 0.90),
        "p95": percentile(items, 0.95),
        "maximum": max(items) if items else None,
    }


def atomic_write_text(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing output: {path}; use --force"
        )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())

    temporary_path.replace(path)


def selector_config_to_dict(config: AnchorSelectorConfig) -> dict[str, Any]:
    return {
        "camera_names": list(config.camera_names),
        "frame_offsets_ns": list(config.frame_offsets_ns),
        "camera_tolerance_ns": config.camera_tolerance_ns,
        "prefer_exact_camera_sync": config.prefer_exact_camera_sync,
        "ego_history_points": config.ego_history_points,
        "ego_history_interval_ns": config.ego_history_interval_ns,
        "ego_history_span_ns": config.ego_history_span_ns,
        "route_maximum_age_ns": config.route_maximum_age_ns,
        "actor_tolerance_ns": config.actor_tolerance_ns,
        "future_horizon_ns": config.future_horizon_ns,
        "minimum_spacing_ns": config.minimum_spacing_ns,
    }


def selected_anchor_record(
    result: AnchorSelectionResult,
    selected_index: int,
    anchor: Any,
    config: AnchorSelectorConfig,
) -> dict[str, Any]:
    return {
        "anchor_format_version": ANCHOR_FORMAT_VERSION,
        "anchor_selector_version": SCRIPT_VERSION,
        "anchor_id": f"{result.clip_id}_{anchor.anchor_ns}",
        "clip_id": result.clip_id,
        "anchor_ns": anchor.anchor_ns,
        "selected_index_within_clip": selected_index,
        "anchor_source": "shared_current_camera_timestamp",
        "visual_profile": "four_camera_three_frame_v0.1",
        "frame_offsets_ns": list(config.frame_offsets_ns),
        "visual_sequence_mode": anchor.visual_sequence_mode,
        "visual_group_modes": list(anchor.visual_group_modes),
        "maximum_camera_skew_ns": anchor.maximum_camera_skew_ns,
        "maximum_camera_target_error_ns": (
            anchor.maximum_camera_target_error_ns
        ),
        "ego_history_profile": "ego_history_16x10hz_v0.1",
        "route_time_error_ns": anchor.route_time_error_ns,
        "actor_time_error_ns": anchor.actor_time_error_ns,
        "future_horizon_ns": config.future_horizon_ns,
        "checks": dict(anchor.checks),
        "failure_reasons": list(anchor.failure_reasons),
        "fully_valid": anchor.fully_valid,
        "minimum_spacing_ns": config.minimum_spacing_ns,
    }


def per_clip_record(
    result: AnchorSelectionResult,
    duration_sec: float,
) -> dict[str, Any]:
    failure_counts: Counter[str] = Counter()
    sequence_modes: Counter[str] = Counter()
    group_modes: Counter[str] = Counter()
    maximum_skews: list[int] = []
    maximum_target_errors: list[int] = []
    route_errors: list[int] = []
    actor_errors: list[int] = []

    for evaluation in result.evaluations:
        failure_counts.update(evaluation.failure_reasons)
        if evaluation.visual_sequence_mode is not None:
            sequence_modes[evaluation.visual_sequence_mode] += 1
        group_modes.update(evaluation.visual_group_modes)
        if evaluation.maximum_camera_skew_ns is not None:
            maximum_skews.append(evaluation.maximum_camera_skew_ns)
        if evaluation.maximum_camera_target_error_ns is not None:
            maximum_target_errors.append(
                evaluation.maximum_camera_target_error_ns
            )
        if evaluation.route_time_error_ns is not None:
            route_errors.append(abs(evaluation.route_time_error_ns))
        if evaluation.actor_time_error_ns is not None:
            actor_errors.append(abs(evaluation.actor_time_error_ns))

    return {
        "clip_id": result.clip_id,
        "duration_sec": duration_sec,
        "source_candidate_count": result.source_candidate_count,
        "boundary_eligible_count": result.boundary_eligible_count,
        "fully_valid_count": result.fully_valid_count,
        "selected_count": result.selected_count,
        "fully_valid_ratio": (
            result.fully_valid_count / result.boundary_eligible_count
            if result.boundary_eligible_count
            else 0.0
        ),
        "selected_ratio": (
            result.selected_count / result.boundary_eligible_count
            if result.boundary_eligible_count
            else 0.0
        ),
        "selected_anchor_ns": [
            anchor.anchor_ns for anchor in result.selected_anchors
        ],
        "failure_counts": dict(failure_counts),
        "visual_sequence_modes": dict(sequence_modes),
        "visual_group_modes": dict(group_modes),
        "maximum_camera_skew_ns": max(maximum_skews, default=None),
        "maximum_camera_target_error_ns": max(
            maximum_target_errors,
            default=None,
        ),
        "maximum_absolute_route_time_error_ns": max(
            route_errors,
            default=None,
        ),
        "maximum_absolute_actor_time_error_ns": max(
            actor_errors,
            default=None,
        ),
    }


def aggregate_summary(
    clip_records: list[dict[str, Any]],
    selected_anchor_count: int,
    processing_errors: list[dict[str, str]],
    config: AnchorSelectorConfig,
    manifest_path: Path,
    elapsed_sec: float,
) -> dict[str, Any]:
    failure_counts: Counter[str] = Counter()
    sequence_modes: Counter[str] = Counter()
    group_modes: Counter[str] = Counter()

    for record in clip_records:
        failure_counts.update(record["failure_counts"])
        sequence_modes.update(record["visual_sequence_modes"])
        group_modes.update(record["visual_group_modes"])

    source_counts = [
        float(record["source_candidate_count"]) for record in clip_records
    ]
    eligible_counts = [
        float(record["boundary_eligible_count"]) for record in clip_records
    ]
    valid_counts = [
        float(record["fully_valid_count"]) for record in clip_records
    ]
    selected_counts = [
        float(record["selected_count"]) for record in clip_records
    ]

    total_source = sum(record["source_candidate_count"] for record in clip_records)
    total_eligible = sum(
        record["boundary_eligible_count"] for record in clip_records
    )
    total_valid = sum(record["fully_valid_count"] for record in clip_records)

    zero_selected = [
        record["clip_id"]
        for record in clip_records
        if record["selected_count"] == 0
    ]

    return {
        "anchor_format_version": ANCHOR_FORMAT_VERSION,
        "anchor_selector_version": SCRIPT_VERSION,
        "generated_at": utc_now_iso(),
        "input_manifest": str(manifest_path),
        "elapsed_sec": elapsed_sec,
        "configuration": selector_config_to_dict(config),
        "clips": {
            "attempted": len(clip_records) + len(processing_errors),
            "processed": len(clip_records),
            "processing_error_count": len(processing_errors),
            "with_selected_anchors": len(clip_records) - len(zero_selected),
            "without_selected_anchors": len(zero_selected),
            "zero_selected_clip_ids": zero_selected,
            "processing_errors": processing_errors,
        },
        "anchors": {
            "source_candidate_total": total_source,
            "boundary_eligible_total": total_eligible,
            "fully_valid_total": total_valid,
            "selected_total": selected_anchor_count,
            "fully_valid_ratio_of_boundary_eligible": (
                total_valid / total_eligible if total_eligible else 0.0
            ),
            "selected_ratio_of_boundary_eligible": (
                selected_anchor_count / total_eligible
                if total_eligible
                else 0.0
            ),
            "selected_ratio_of_fully_valid": (
                selected_anchor_count / total_valid if total_valid else 0.0
            ),
            "source_candidates_per_clip": distribution(source_counts),
            "boundary_eligible_per_clip": distribution(eligible_counts),
            "fully_valid_per_clip": distribution(valid_counts),
            "selected_per_clip": distribution(selected_counts),
        },
        "failure_counts": dict(failure_counts),
        "visual_sequence_modes": dict(sequence_modes),
        "visual_group_modes": dict(group_modes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build camera-driven candidate anchors for AlpaSim clips."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--anchor-output",
        type=Path,
        default=DEFAULT_ANCHOR_OUTPUT,
    )
    parser.add_argument(
        "--per-clip-output",
        type=Path,
        default=DEFAULT_PER_CLIP_OUTPUT,
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
    )
    parser.add_argument(
        "--limit-clips",
        type=int,
        default=None,
        help="Process only the first N eligible clips for a smoke test.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()

    manifest_records = read_jsonl(manifest_path)
    eligible_records = [
        record
        for record in manifest_records
        if record.get("manifest_usable") is True
    ]

    if args.limit_clips is not None:
        if args.limit_clips <= 0:
            raise ValueError("--limit-clips must be positive")
        eligible_records = eligible_records[: args.limit_clips]

    config = AnchorSelectorConfig()
    selector = AnchorSelector(config)
    anchor_records: list[dict[str, Any]] = []
    clip_records: list[dict[str, Any]] = []
    processing_errors: list[dict[str, str]] = []

    started = time.monotonic()
    total = len(eligible_records)

    for position, manifest_record in enumerate(eligible_records, start=1):
        clip_id = str(manifest_record.get("clip_id", ""))
        clip_path = dataset_root / str(manifest_record["clip_path"])

        try:
            reader = DrivingClipReader(clip_path)
            selection = selector.select(reader)
            clip_records.append(
                per_clip_record(
                    selection,
                    duration_sec=reader.duration_ns / 1e9,
                )
            )
            for selected_index, anchor in enumerate(
                selection.selected_anchors
            ):
                anchor_records.append(
                    selected_anchor_record(
                        selection,
                        selected_index,
                        anchor,
                        config,
                    )
                )
        except Exception as exc:
            processing_errors.append(
                {
                    "clip_id": clip_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )

        if position == 1 or position % 25 == 0 or position == total:
            print(f"Processed {position}/{total}: {clip_id}")

    elapsed_sec = time.monotonic() - started
    summary = aggregate_summary(
        clip_records,
        selected_anchor_count=len(anchor_records),
        processing_errors=processing_errors,
        config=config,
        manifest_path=manifest_path,
        elapsed_sec=elapsed_sec,
    )

    anchor_text = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for record in anchor_records
    )
    per_clip_text = "".join(
        json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        + "\n"
        for record in clip_records
    )
    summary_text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"

    atomic_write_text(args.anchor_output, anchor_text, args.force)
    atomic_write_text(args.per_clip_output, per_clip_text, args.force)
    atomic_write_text(args.summary_output, summary_text, args.force)

    anchor_digest = hashlib.sha256(anchor_text.encode("utf-8")).hexdigest()
    print("Anchor output:", args.anchor_output)
    print("Per-clip report:", args.per_clip_output)
    print("Summary report:", args.summary_output)
    print("Anchor SHA-256:", anchor_digest)
    print("Clips processed:", summary["clips"]["processed"])
    print("Processing errors:", summary["clips"]["processing_error_count"])
    print("Boundary-eligible anchors:", summary["anchors"]["boundary_eligible_total"])
    print("Fully valid anchors:", summary["anchors"]["fully_valid_total"])
    print("Selected anchors:", summary["anchors"]["selected_total"])
    print("Failure counts:", summary["failure_counts"])
    print("Visual sequence modes:", summary["visual_sequence_modes"])

    return 1 if processing_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
