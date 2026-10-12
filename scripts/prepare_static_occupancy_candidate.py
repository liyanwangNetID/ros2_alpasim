#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_REPO = "nvidia/PhysicalAI-Autonomous-Vehicles-NuRec"
REQUIRED = ("metadata.yaml", "mesh.ply", "mesh_ground.ply")


def select_artifact(catalog: Path, uuid: str) -> tuple[list[str], dict[str, str]]:
    with catalog.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        matches = [r for r in reader if (r.get("uuid") or "").strip() == uuid]
    if len(matches) != 1:
        raise ValueError(f"Expected one catalog row for UUID {uuid}, found {len(matches)}")
    return fieldnames, {k: (v or "").strip() for k, v in matches[0].items()}


def discover_download(row: dict[str, str]) -> tuple[str, str, str]:
    usdz_values = [v for v in row.values() if v.lower().endswith(".usdz")]
    if len(usdz_values) != 1:
        raise ValueError(f"Could not uniquely discover USDZ path; candidates={usdz_values}")
    filename = usdz_values[0]
    revision = next(
        (row[k] for k in ("hf_revision", "revision", "huggingface_revision") if row.get(k)),
        None,
    )
    if not revision:
        raise ValueError("Catalog row has no Hugging Face revision")
    repo = next(
        (row[k] for k in ("hf_repo_id", "repo_id", "artifact_repository_id") if row.get(k) and "/" in row[k]),
        DEFAULT_REPO,
    )
    return repo, revision, filename


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--scene-cache", type=Path, required=True)
    parser.add_argument("--candidate-catalog", type=Path, required=True)
    parser.add_argument("--result-json", type=Path, required=True)
    args = parser.parse_args()
    fields, row = select_artifact(args.catalog, args.uuid)
    repo, revision, filename = discover_download(row)
    downloaded = Path(hf_hub_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        filename=filename,
        local_dir=args.scene_cache / "hf-download",
    ))
    with zipfile.ZipFile(downloaded, "r") as archive:
        names = set(archive.namelist())
    missing = [name for name in REQUIRED if name not in names]
    result = {
        "uuid": args.uuid,
        "repo_id": repo,
        "revision": revision,
        "filename": filename,
        "downloaded_path": str(downloaded),
        "required_members": list(REQUIRED),
        "missing_members": missing,
        "supports_static_occupancy": not missing,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if missing:
        print(json.dumps(result, ensure_ascii=False))
        return 2
    target = args.scene_cache / "all-usdzs" / f"{args.uuid}.usdz"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, target)
    args.candidate_catalog.parent.mkdir(parents=True, exist_ok=True)
    with args.candidate_catalog.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
