#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    name: str
    release: str
    suite_id: str
    priority: int
    suites_csv: Path
    scenes_csv: Path


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [{k.strip(): (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]


def load(source: Source) -> list[dict[str, str]]:
    artifacts = rows(source.scenes_csv)
    by_uuid = {r["uuid"]: r for r in artifacts if r.get("uuid")}
    by_scene: dict[str, list[dict[str, str]]] = {}
    for artifact in artifacts:
        if artifact.get("scene_id"):
            by_scene.setdefault(artifact["scene_id"], []).append(artifact)
    output = []
    seen = set()
    for suite in rows(source.suites_csv):
        if suite.get("test_suite_id") != source.suite_id:
            continue
        scene_id = suite.get("scene_id", "")
        requested_uuid = suite.get("uuid", "")
        if requested_uuid:
            artifact = by_uuid.get(requested_uuid)
        else:
            candidates = by_scene.get(scene_id, [])
            artifact = candidates[0] if len(candidates) == 1 else None
        if not scene_id or artifact is None:
            raise ValueError(
                f"Cannot resolve exact artifact for scene={scene_id!r} in {source.name}"
            )
        uuid = artifact["uuid"]
        key = (scene_id, uuid)
        if key in seen:
            continue
        seen.add(key)
        output.append({
            "scene_id": scene_id,
            "uuid": uuid,
            "release": source.release,
            "source_name": source.name,
            "source_suite_id": source.suite_id,
            "source_priority": str(source.priority),
            "scenes_csv": str(source.scenes_csv.resolve()),
            "suites_csv": str(source.suites_csv.resolve()),
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-dir", type=Path, default=Path("/home/lab/alpasim/data/scenes"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/lab/alpasim/data/scenes/static_occupancy_scene_candidates.csv"),
    )
    args = parser.parse_args()
    d = args.scenes_dir.resolve()
    sources = [
        Source("latest_public_2604", "2604", "public_2604", 1, d / "sim_suites_latest.csv", d / "sim_scenes_latest.csv"),
        Source("latest_public_2601", "2601", "public_2601", 2, d / "sim_suites_latest.csv", d / "sim_scenes_latest.csv"),
        Source("local_public_2601", "2601_local", "public_2601", 3, d / "sim_suites.csv", d / "sim_scenes.csv"),
        Source("legacy_public_2507", "2507", "public_2507", 4, d / "sim_suites_2505.csv", d / "sim_scenes_2505.csv"),
    ]
    candidates_by_scene: dict[str, list[dict[str, str]]] = {}
    seen_artifacts = set()
    for source in sources:
        loaded = load(source)
        print(f"Loaded {len(loaded)} candidates from {source.name}")
        for candidate in loaded:
            artifact_key = (candidate["scene_id"], candidate["uuid"])
            if artifact_key in seen_artifacts:
                continue
            seen_artifacts.add(artifact_key)
            candidates_by_scene.setdefault(candidate["scene_id"], []).append(candidate)
    scene_ids = sorted(
        candidates_by_scene,
        key=lambda scene: (
            min(int(c["source_priority"]) for c in candidates_by_scene[scene]),
            scene,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_index", "candidate_rank", "candidate_count", "scene_id", "uuid",
        "release", "source_name", "source_suite_id", "source_priority",
        "scenes_csv", "suites_csv",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene_index, scene_id in enumerate(scene_ids, 1):
            candidates = sorted(
                candidates_by_scene[scene_id],
                key=lambda c: (int(c["source_priority"]), c["uuid"]),
            )
            for rank, candidate in enumerate(candidates, 1):
                writer.writerow({
                    "scene_index": scene_index,
                    "candidate_rank": rank,
                    "candidate_count": len(candidates),
                    **candidate,
                })
    multi = sum(len(v) > 1 for v in candidates_by_scene.values())
    print(f"Output: {args.output}")
    print(f"Unique scenes: {len(scene_ids)}")
    print(f"Artifact candidates: {sum(map(len, candidates_by_scene.values()))}")
    print(f"Scenes with historical fallbacks: {multi}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
