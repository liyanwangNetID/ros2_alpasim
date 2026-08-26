#!/usr/bin/env python3

"""Generate a deduplicated manifest of all available AlpaSim scenes.

The script combines the following catalogs:

1. Latest public_2604
2. Latest public_2601
3. Local older public_2601
4. Legacy public_2507 from the 2505 CSV files

Scenes are deduplicated globally by scene_id.

If the same logical scene exists in multiple releases, only the
highest-priority artifact is retained:

    public_2604
    latest public_2601
    local public_2601
    public_2507 / 2505

The generated manifest records the exact UUID and the sim_scenes CSV
that must be used to resolve and download that artifact.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CatalogSource:
    """Describe one suite and artifact catalog pair."""

    name: str
    release: str
    suite_id: str
    priority: int
    suites_csv: Path
    scenes_csv: Path


@dataclass(frozen=True)
class SceneArtifact:
    """One selected logical scene and its exact USDZ artifact."""

    scene_id: str
    uuid: str
    release: str
    source_name: str
    source_suite_id: str
    priority: int
    scenes_csv: Path
    suites_csv: Path


def read_csv_rows(
    path: Path,
) -> list[dict[str, str]]:
    """Read a CSV file and normalize whitespace in all fields."""

    if not path.is_file():
        raise FileNotFoundError(
            f"CSV file does not exist: {path}"
        )

    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV file has no header: {path}"
            )

        rows: list[dict[str, str]] = []

        for raw_row in reader:
            row = {
                str(key).strip(): (
                    str(value).strip()
                    if value is not None
                    else ""
                )
                for key, value in raw_row.items()
                if key is not None
            }

            rows.append(row)

    return rows


def require_columns(
    path: Path,
    rows: list[dict[str, str]],
    required_columns: set[str],
) -> None:
    """Validate that a CSV contains the required columns."""

    if not rows:
        raise ValueError(
            f"CSV file contains no data rows: {path}"
        )

    available_columns = set(rows[0])

    missing_columns = (
        required_columns - available_columns
    )

    if missing_columns:
        raise ValueError(
            f"CSV file {path} is missing columns: "
            f"{sorted(missing_columns)}; "
            f"available columns: "
            f"{sorted(available_columns)}"
        )


def build_artifact_index(
    scenes_csv: Path,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, list[dict[str, str]]],
]:
    """Index artifact metadata by UUID and scene ID."""

    rows = read_csv_rows(
        scenes_csv
    )

    require_columns(
        scenes_csv,
        rows,
        {
            "uuid",
            "scene_id",
        },
    )

    by_uuid: dict[
        str,
        dict[str, str],
    ] = {}

    by_scene_id: dict[
        str,
        list[dict[str, str]],
    ] = {}

    for row in rows:
        uuid = row["uuid"]
        scene_id = row["scene_id"]

        if not uuid or not scene_id:
            continue

        if uuid in by_uuid:
            previous = by_uuid[uuid]

            if (
                previous["scene_id"]
                != scene_id
            ):
                raise ValueError(
                    "The same UUID is associated with "
                    "different scene IDs in "
                    f"{scenes_csv}: "
                    f"{uuid}"
                )

        by_uuid[uuid] = row

        by_scene_id.setdefault(
            scene_id,
            [],
        ).append(row)

    return by_uuid, by_scene_id


def resolve_scene_uuid(
    *,
    scene_id: str,
    requested_uuid: str,
    scenes_csv: Path,
    by_uuid: dict[str, dict[str, str]],
    by_scene_id: dict[
        str,
        list[dict[str, str]],
    ],
) -> str:
    """Resolve and validate the UUID for one suite row."""

    if requested_uuid:
        artifact = by_uuid.get(
            requested_uuid
        )

        if artifact is None:
            raise ValueError(
                f"UUID {requested_uuid!r} from "
                f"the suite was not found in "
                f"{scenes_csv}"
            )

        artifact_scene_id = (
            artifact["scene_id"]
        )

        if artifact_scene_id != scene_id:
            raise ValueError(
                f"UUID {requested_uuid!r} maps to "
                f"scene {artifact_scene_id!r}, "
                f"not requested scene "
                f"{scene_id!r}"
            )

        return requested_uuid

    candidates = by_scene_id.get(
        scene_id,
        [],
    )

    if not candidates:
        raise ValueError(
            f"Scene {scene_id!r} was not found "
            f"in {scenes_csv}"
        )

    if len(candidates) == 1:
        return candidates[0]["uuid"]

    candidate_uuids = sorted(
        {
            candidate["uuid"]
            for candidate in candidates
        }
    )

    raise ValueError(
        f"Scene {scene_id!r} has multiple "
        f"artifacts in {scenes_csv}, but the "
        "suite row does not specify a UUID: "
        f"{candidate_uuids}"
    )


def load_catalog_source(
    source: CatalogSource,
) -> list:
    """Load all selected scenes from one catalog source."""

    suite_rows = read_csv_rows(
        source.suites_csv
    )

    require_columns(
        source.suites_csv,
        suite_rows,
        {
            "test_suite_id",
            "scene_id",
        },
    )

    by_uuid, by_scene_id = (
        build_artifact_index(
            source.scenes_csv
        )
    )

    selected: list[
        SceneArtifact
    ] = []

    seen_in_source: set[str] = set()

    for row in suite_rows:
        if (
            row["test_suite_id"]
            != source.suite_id
        ):
            continue

        scene_id = row["scene_id"]

        if not scene_id:
            continue

        if scene_id in seen_in_source:
            raise ValueError(
                f"Duplicate scene_id {scene_id!r} "
                f"inside suite "
                f"{source.suite_id!r} in "
                f"{source.suites_csv}"
            )

        requested_uuid = row.get(
            "uuid",
            "",
        )

        uuid = resolve_scene_uuid(
            scene_id=scene_id,
            requested_uuid=requested_uuid,
            scenes_csv=(
                source.scenes_csv
            ),
            by_uuid=by_uuid,
            by_scene_id=by_scene_id,
        )

        selected.append(
            SceneArtifact(
                scene_id=scene_id,
                uuid=uuid,
                release=source.release,
                source_name=source.name,
                source_suite_id=(
                    source.suite_id
                ),
                priority=source.priority,
                scenes_csv=(
                    source.scenes_csv.resolve()
                ),
                suites_csv=(
                    source.suites_csv.resolve()
                ),
            )
        )

        seen_in_source.add(
            scene_id
        )

    if not selected:
        raise ValueError(
            f"Suite {source.suite_id!r} "
            f"was not found or contained no "
            f"scenes in {source.suites_csv}"
        )

    return selected


def deduplicate_sources(
    sources: list[CatalogSource],
) -> tuple[
    list[SceneArtifact],
    dict[str, int],
]:
    """Combine sources and retain the highest-priority artifact."""

    selected_by_scene_id: dict[
        str,
        SceneArtifact,
    ] = {}

    replaced_counts: dict[
        str,
        int,
    ] = {
        source.name: 0
        for source in sources
    }

    for source in sorted(
        sources,
        key=lambda item: item.priority,
    ):
        artifacts = load_catalog_source(
            source
        )

        print(
            f"Loaded {len(artifacts):4d} scenes "
            f"from {source.name}"
        )

        for artifact in artifacts:
            existing = (
                selected_by_scene_id.get(
                    artifact.scene_id
                )
            )

            if existing is None:
                selected_by_scene_id[
                    artifact.scene_id
                ] = artifact
                continue

            # Sources are processed in priority order.
            # The existing artifact therefore has equal
            # or higher priority and is retained.
            replaced_counts[
                source.name
            ] += 1

    selected = sorted(
        selected_by_scene_id.values(),
        key=lambda item: (
            item.priority,
            item.scene_id,
        ),
    )

    return selected, replaced_counts


def write_manifest(
    output_path: Path,
    scenes: list[SceneArtifact],
) -> None:
    """Write the final unique-scene manifest."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "index",
        "scene_id",
        "uuid",
        "release",
        "source_name",
        "source_suite_id",
        "scenes_csv",
        "suites_csv",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index, scene in enumerate(
            scenes,
            start=1,
        ):
            writer.writerow(
                {
                    "index": index,
                    "scene_id": (
                        scene.scene_id
                    ),
                    "uuid": scene.uuid,
                    "release": (
                        scene.release
                    ),
                    "source_name": (
                        scene.source_name
                    ),
                    "source_suite_id": (
                        scene.source_suite_id
                    ),
                    "scenes_csv": str(
                        scene.scenes_csv
                    ),
                    "suites_csv": str(
                        scene.suites_csv
                    ),
                }
            )


def print_summary(
    scenes: list[SceneArtifact],
    sources: list[CatalogSource],
    skipped_counts: dict[str, int],
    output_path: Path,
) -> None:
    """Print a summary of the generated manifest."""

    release_counts: dict[
        str,
        int,
    ] = {}

    source_counts: dict[
        str,
        int,
    ] = {}

    for scene in scenes:
        release_counts[
            scene.release
        ] = (
            release_counts.get(
                scene.release,
                0,
            )
            + 1
        )

        source_counts[
            scene.source_name
        ] = (
            source_counts.get(
                scene.source_name,
                0,
            )
            + 1
        )

    print()
    print(
        "=" * 64
    )
    print(
        "Unique AlpaSim scene manifest generated"
    )
    print(
        "=" * 64
    )
    print(
        f"Output: {output_path.resolve()}"
    )
    print(
        f"Unique scene IDs: {len(scenes)}"
    )

    print()
    print(
        "Selected artifacts by source:"
    )

    for source in sorted(
        sources,
        key=lambda item: item.priority,
    ):
        selected_count = (
            source_counts.get(
                source.name,
                0,
            )
        )

        skipped_count = (
            skipped_counts.get(
                source.name,
                0,
            )
        )

        print(
            f"  {source.name}: "
            f"selected={selected_count}, "
            f"skipped_as_duplicate="
            f"{skipped_count}"
        )

    print()
    print(
        "Selected artifacts by release:"
    )

    for release, count in sorted(
        release_counts.items()
    ):
        print(
            f"  {release}: {count}"
        )


def parse_arguments() -> (
    argparse.Namespace
):
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a globally deduplicated "
            "AlpaSim scene manifest."
        )
    )

    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=Path(
            "/home/lab/alpasim/"
            "data/scenes"
        ),
        help=(
            "Directory containing the "
            "sim_scenes and sim_suites CSV files."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output manifest path. Defaults to "
            "<scenes-dir>/all_unique_scenes.csv."
        ),
    )

    return parser.parse_args()


def main() -> int:
    """Generate the unique-scene manifest."""

    arguments = parse_arguments()

    scenes_dir = (
        arguments.scenes_dir.resolve()
    )

    output_path = (
        arguments.output.resolve()
        if arguments.output is not None
        else (
            scenes_dir
            / "all_unique_scenes.csv"
        )
    )

    # Lower number means higher priority.
    sources = [
        CatalogSource(
            name="latest_public_2604",
            release="2604",
            suite_id="public_2604",
            priority=1,
            suites_csv=(
                scenes_dir
                / "sim_suites_latest.csv"
            ),
            scenes_csv=(
                scenes_dir
                / "sim_scenes_latest.csv"
            ),
        ),
        CatalogSource(
            name="latest_public_2601",
            release="2601",
            suite_id="public_2601",
            priority=2,
            suites_csv=(
                scenes_dir
                / "sim_suites_latest.csv"
            ),
            scenes_csv=(
                scenes_dir
                / "sim_scenes_latest.csv"
            ),
        ),
        CatalogSource(
            name="local_public_2601",
            release="2601_local",
            suite_id="public_2601",
            priority=3,
            suites_csv=(
                scenes_dir
                / "sim_suites.csv"
            ),
            scenes_csv=(
                scenes_dir
                / "sim_scenes.csv"
            ),
        ),
        CatalogSource(
            name="legacy_public_2507",
            release="2507",
            suite_id="public_2507",
            priority=4,
            suites_csv=(
                scenes_dir
                / "sim_suites_2505.csv"
            ),
            scenes_csv=(
                scenes_dir
                / "sim_scenes_2505.csv"
            ),
        ),
    ]

    try:
        scenes, skipped_counts = (
            deduplicate_sources(
                sources
            )
        )

        if not scenes:
            raise RuntimeError(
                "No scenes were selected"
            )

        scene_ids = [
            scene.scene_id
            for scene in scenes
        ]

        if (
            len(scene_ids)
            != len(set(scene_ids))
        ):
            raise RuntimeError(
                "Internal error: duplicate "
                "scene IDs remain after "
                "deduplication"
            )

        write_manifest(
            output_path,
            scenes,
        )

        print_summary(
            scenes,
            sources,
            skipped_counts,
            output_path,
        )

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())