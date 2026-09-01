#!/usr/bin/env python3
"""Scan conservative longitudinal Meta-action thresholds.

The tool reads meta_action_features_v0.1.jsonl only. It compares candidate
rules for stop, accelerate, decelerate, and maintain_speed. It does not write
final labels.

Priority for every threshold combination:
1. stop, using terminal speed plus sustained low-speed duration;
2. accelerate/decelerate, requiring agreement between total speed change and
   first-half/second-half mean-speed change;
3. maintain_speed fallback;
4. conflicting signs are reported separately but conservatively fall back to
   maintain_speed in the candidate counts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_VERSION = "0.1.0"
SCAN_FORMAT_VERSION = "0.1-draft"
from project_paths import ALPASIM_DATA_ROOT

ROOT = ALPASIM_DATA_ROOT
DEFAULT_INPUT = (
    ROOT / "annotations" / "v0.1-draft" / "intermediate"
    / "meta_action_features_v0.1.jsonl"
)
DEFAULT_SUMMARY = ROOT / "reports" / "longitudinal_threshold_scan_v0.1.json"
DEFAULT_CASES = ROOT / "reports" / "longitudinal_threshold_scan_cases_v0.1.jsonl"

DEFAULT_STOP_SPEEDS_MPS = (0.1, 0.3, 0.5)
DEFAULT_STOP_DURATIONS_SEC = (0.5, 1.0, 1.5, 2.0)
DEFAULT_SPEED_DELTAS_MPS = (0.5, 1.0, 1.5, 2.0)
DEFAULT_HALF_MEAN_DELTAS_MPS = (0.25, 0.5, 0.75, 1.0)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def atomic_write(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise FileExistsError(f"output exists: {path}; use --force")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=path.name + ".", suffix=".tmp", delete=False,
    ) as file:
        temporary = Path(file.name)
        file.write(content)
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def parse_values(text: str) -> tuple[float, ...]:
    values = tuple(sorted(set(float(part.strip()) for part in text.split(",") if part.strip())))
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("threshold values must be positive")
    return values


def low_speed_duration_key(stop_speed_mps: float) -> str:
    mapping = {
        0.1: "longest_duration_below_0_3_mps_sec",
        0.3: "longest_duration_below_0_3_mps_sec",
        0.5: "longest_duration_below_0_5_mps_sec",
    }
    try:
        return mapping[round(stop_speed_mps, 1)]
    except KeyError as exc:
        raise ValueError(
            "stop speed must be one of 0.1, 0.3, or 0.5 m/s because the "
            "current feature file only contains 0.3 and 0.5 duration bands"
        ) from exc


def longitudinal_metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return record["longitudinal"]


def classify(
    record: Mapping[str, Any],
    *,
    stop_speed_mps: float,
    stop_duration_sec: float,
    speed_delta_mps: float,
    half_mean_delta_mps: float,
) -> tuple[str, list[str]]:
    values = longitudinal_metrics(record)
    duration_key = low_speed_duration_key(stop_speed_mps)
    reasons: list[str] = []

    if (
        float(values["final_speed_mps"]) <= stop_speed_mps
        and float(values[duration_key]) >= stop_duration_sec
    ):
        return "stop", ["terminal_low_speed", "sustained_low_speed"]

    total_delta = float(values["speed_delta_mps"])
    half_delta = float(values["second_half_minus_first_half_mean_speed_mps"])
    if total_delta >= speed_delta_mps and half_delta >= half_mean_delta_mps:
        return "accelerate", ["positive_total_speed_change", "positive_half_mean_change"]
    if total_delta <= -speed_delta_mps and half_delta <= -half_mean_delta_mps:
        return "decelerate", ["negative_total_speed_change", "negative_half_mean_change"]

    if total_delta * half_delta < 0.0:
        reasons.append("conflicting_speed_change_signs")
    elif abs(total_delta) >= speed_delta_mps:
        reasons.append("total_change_only")
    elif abs(half_delta) >= half_mean_delta_mps:
        reasons.append("half_mean_change_only")
    else:
        reasons.append("changes_below_thresholds")
    return "maintain_speed", reasons


def evaluate(
    records: Sequence[Mapping[str, Any]],
    *,
    stop_speed_mps: float,
    stop_duration_sec: float,
    speed_delta_mps: float,
    half_mean_delta_mps: float,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    ids: dict[str, list[str]] = {
        "stop": [],
        "accelerate": [],
        "decelerate": [],
        "maintain_speed": [],
        "conflict": [],
    }
    for record in records:
        label, reasons = classify(
            record,
            stop_speed_mps=stop_speed_mps,
            stop_duration_sec=stop_duration_sec,
            speed_delta_mps=speed_delta_mps,
            half_mean_delta_mps=half_mean_delta_mps,
        )
        anchor_id = str(record["anchor_id"])
        counts[label] += 1
        ids[label].append(anchor_id)
        for reason in reasons:
            reason_counts[reason] += 1
        if "conflicting_speed_change_signs" in reasons:
            counts["conflicting_signs"] += 1
            ids["conflict"].append(anchor_id)
    return {
        "stop_speed_mps": stop_speed_mps,
        "stop_duration_sec": stop_duration_sec,
        "speed_delta_mps": speed_delta_mps,
        "half_mean_delta_mps": half_mean_delta_mps,
        "counts": dict(counts),
        "reason_counts": dict(reason_counts),
        "anchor_ids": ids,
    }


def sample(values: Sequence[str], count: int, seed: int) -> list[str]:
    if len(values) <= count:
        return list(values)
    generator = random.Random(seed)
    return sorted(generator.sample(list(values), count))


def case_record(
    record: Mapping[str, Any],
    source: str,
    label: str,
    rule: Mapping[str, float],
) -> dict[str, Any]:
    values = longitudinal_metrics(record)
    return {
        "scan_format_version": SCAN_FORMAT_VERSION,
        "scanner_version": SCRIPT_VERSION,
        "source": source,
        "candidate_label": label,
        "anchor_id": record["anchor_id"],
        "clip_id": record["clip_id"],
        "anchor_ns": record["anchor_ns"],
        "rule": dict(rule),
        "metrics": {
            key: values[key]
            for key in (
                "initial_speed_mps",
                "final_speed_mps",
                "minimum_speed_mps",
                "maximum_speed_mps",
                "mean_speed_mps",
                "speed_delta_mps",
                "derived_mean_acceleration_mps2",
                "second_half_minus_first_half_mean_speed_mps",
                "longest_duration_below_0_3_mps_sec",
                "longest_duration_below_0_5_mps_sec",
            )
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--case-output", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--stop-speeds-mps", default="0.1,0.3,0.5")
    parser.add_argument("--stop-durations-sec", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--speed-deltas-mps", default="0.5,1.0,1.5,2.0")
    parser.add_argument("--half-mean-deltas-mps", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--samples-per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_group <= 0:
        raise ValueError("--samples-per-group must be positive")
    records = read_jsonl(args.feature_input.expanduser().resolve())
    stop_speeds = parse_values(args.stop_speeds_mps)
    stop_durations = parse_values(args.stop_durations_sec)
    speed_deltas = parse_values(args.speed_deltas_mps)
    half_deltas = parse_values(args.half_mean_deltas_mps)

    rows = [
        evaluate(
            records,
            stop_speed_mps=stop_speed,
            stop_duration_sec=stop_duration,
            speed_delta_mps=speed_delta,
            half_mean_delta_mps=half_delta,
        )
        for stop_speed, stop_duration, speed_delta, half_delta in itertools.product(
            stop_speeds, stop_durations, speed_deltas, half_deltas
        )
    ]

    by_id = {str(record["anchor_id"]): record for record in records}
    cases = []
    compact_rows = []
    for index, row in enumerate(rows):
        rule = {
            "stop_speed_mps": float(row["stop_speed_mps"]),
            "stop_duration_sec": float(row["stop_duration_sec"]),
            "speed_delta_mps": float(row["speed_delta_mps"]),
            "half_mean_delta_mps": float(row["half_mean_delta_mps"]),
        }
        source = (
            f"stop_{rule['stop_speed_mps']:g}_{rule['stop_duration_sec']:g}_"
            f"motion_{rule['speed_delta_mps']:g}_{rule['half_mean_delta_mps']:g}"
        )
        for label in ("stop", "accelerate", "decelerate", "maintain_speed", "conflict"):
            selected = sample(
                row["anchor_ids"][label],
                args.samples_per_group,
                args.seed + index * 10 + len(cases),
            )
            for anchor_id in selected:
                cases.append(case_record(by_id[anchor_id], source, label, rule))
        compact = dict(row)
        compact.pop("anchor_ids")
        compact_rows.append(compact)

    summary = {
        "scan_format_version": SCAN_FORMAT_VERSION,
        "scanner_version": SCRIPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_input": str(args.feature_input),
        "feature_record_count": len(records),
        "classification_priority": [
            "stop",
            "accelerate_or_decelerate_with_two_signal_agreement",
            "maintain_speed_fallback",
        ],
        "configuration": {
            "stop_speeds_mps": list(stop_speeds),
            "stop_durations_sec": list(stop_durations),
            "speed_deltas_mps": list(speed_deltas),
            "half_mean_deltas_mps": list(half_deltas),
        },
        "scan_rows": compact_rows,
    }
    case_text = "".join(
        json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n"
        for value in cases
    )
    atomic_write(
        args.summary_output,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        args.force,
    )
    atomic_write(args.case_output, case_text, args.force)
    print("Feature records:", len(records))
    print("Threshold combinations:", len(rows))
    print("Sampled case records:", len(cases))
    print("Summary output:", args.summary_output)
    print("Case output:", args.case_output)
    print("Case SHA-256:", hashlib.sha256(case_text.encode()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
