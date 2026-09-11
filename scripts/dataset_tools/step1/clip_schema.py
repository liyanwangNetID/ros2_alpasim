"""Generate the frozen Step 1 Clip and Clip Manifest schema document."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from project_paths import SCHEMA_ROOT
from step1.clip_manifest import (
    CAMERA_NAMES,
    JSONL_RELATIVE_PATHS,
    MANIFEST_VERSION,
    REQUIRED_RELATIVE_PATHS,
    SCRIPT_VERSION,
)

DEFAULT_OUTPUT = SCHEMA_ROOT / "clip_schema.md"


def build_clip_schema_markdown() -> str:
    cameras = "\n".join(f"- `{name}`" for name in CAMERA_NAMES)
    required_files = "\n".join(
        f"- `{name}`" for name in REQUIRED_RELATIVE_PATHS
    )
    jsonl_files = "\n".join(
        f"- `{name}`" for name in JSONL_RELATIVE_PATHS
    )
    sections = [
        f"# AlpaSim Clip and Clip Manifest Schema v{MANIFEST_VERSION}",
        "",
        "## Status and versions",
        "",
        f"- Manifest schema version: `{MANIFEST_VERSION}`",
        f"- Manifest builder version: `{SCRIPT_VERSION}`",
        "- Formal manifest: `manifests/clips_v0.1.jsonl`",
        "- Formal summary: `reports/clip_manifest_summary_v0.1.json`",
        "- Time unit: integer nanoseconds unless a field ends in `_sec` or `_ms`",
        "- Encoding: UTF-8 for JSON and JSONL",
        "",
        "Each non-empty manifest line is one JSON object for one finalized Clip.",
        "Rows are ordered by numeric Clip number and `clip_id` is unique.",
        "",
        "## Clip identity and root layout",
        "",
        "A finalized Clip directory is named `test_clip_NNN` and is located",
        "directly below the configured AlpaSim data root. The manifest stores",
        "the directory name as `clip_id`, its number as `clip_number`, and its",
        "path relative to the data root as `clip_path`.",
        "",
        "### Required files",
        "",
        required_files,
        "",
        "The following JSONL files are counted by non-empty line:",
        "",
        jsonl_files,
        "",
        "## Selected cameras",
        "",
        "The frozen camera order is:",
        "",
        cameras,
        "",
        "Each camera has `cameras/<camera_name>/timestamps.jsonl`,",
        "`calibration/<camera_name>.json`, and referenced JPEG files.",
        "Each timestamp row contains integer `stamp_ns` and non-empty",
        "`image_path` relative to the camera directory.",
        "Timestamps must be strictly increasing and unique. Every indexed",
        "image must exist and every JPEG must be indexed.",
        "",
        "## Time and coordinate conventions",
        "",
        "- Simulation and camera timestamp fields ending in `_ns` use integer nanoseconds.",
        "- `sim_duration_sec` uses seconds.",
        "- Camera interval fields ending in `_ms` use milliseconds.",
        "- Step 1 performs no coordinate transformation. Raw coordinate meanings",
        "  remain those of the recorded files, calibration, and VectorMap.",
        "",
        "## Clip manifest row",
        "",
        "Top-level fields:",
        "",
        "- `manifest_version`: constant `0.1`",
        "- `manifest_builder_version`: constant `0.1.0`",
        "- `clip_id`, `clip_number`, `clip_path`",
        "- `dataset_format_version`, `recorder_status`",
        "- `validation_valid`, `manifest_usable`",
        "- `first_sim_time_ns`, `last_sim_time_ns`, `sim_duration_sec`",
        "- `size_bytes`",
        "- `cameras`, `camera_statistics`",
        "- `jsonl_line_counts`, `recorder_topic_counts`",
        "- `validation_checks`, `validation_required_checks`",
        "- `failed_validation_checks`, `derived_checks`",
        "- `missing_required_files`, `errors`",
        "",
        "### Per-camera summary",
        "",
        "Each camera summary contains `row_count`, `first_stamp_ns`,",
        "`last_stamp_ns`, `strictly_increasing`, `unique_timestamps`,",
        "`missing_image_count`, `unindexed_jpeg_count`, `median_interval_ms`,",
        "`maximum_interval_ms`, and `calibration_present`.",
        "",
        "## Derived quality checks",
        "",
        "`derived_checks` contains:",
        "",
        "- `metadata_readable`",
        "- `validation_readable`",
        "- `required_files_present`",
        "- `camera_indexes_readable`",
        "- `camera_images_complete`",
        "- `camera_timestamps_valid`",
        "- `route_available`",
        "- `complete_ground_truth_present`",
        "- `vector_map_present`",
        "",
        "`manifest_usable` is true only when metadata and validation are readable,",
        "all required files exist, all camera indexes and images are complete,",
        "timestamps are valid, complete ground truth and VectorMap are present,",
        "and `validation_valid` is true. `route_available` is reported separately.",
        "",
        "## Aggregate summary",
        "",
        "`reports/clip_manifest_summary_v0.1.json` contains versions, a dynamic",
        "`generated_at`, Clip counts, failure counts, format counts, size and",
        "duration statistics, and per-camera frame-count statistics.",
        "Comparisons exclude `generated_at`. With unchanged source Clips, the",
        "compact JSONL manifest is byte-identical across runs.",
        "",
    ]
    return "\n".join(sections)


def write_clip_schema(
    output_path: Path = DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_path}. Use force."
        )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=output_path.parent,
        prefix=output_path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        file.write(build_clip_schema_markdown())
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(output_path)
    return output_path


def main() -> int:
    path = write_clip_schema(force=True)
    print("Clip schema written:", path)
    print("PASS: Step 1 Clip schema generated deterministically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
