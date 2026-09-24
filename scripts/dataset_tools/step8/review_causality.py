#!/usr/bin/env python3
"""Review one Step 8 Structured CoC record with the Step 7 scene video."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from .causality import build_structured_causality
from .contract import JoinedAnchor, Step8Paths, load_and_validate_inputs


def _matches(
    joined: JoinedAnchor,
    record: dict[str, Any],
    *,
    anchor_id: str | None,
    rule_id: str | None,
    quality: str | None,
) -> bool:
    if anchor_id is not None and joined.anchor_id != anchor_id:
        return False
    if quality is not None and record["quality"]["status"] != quality:
        return False
    if rule_id is not None and not any(
        link["rule_id"] == rule_id
        for link in record["structured_coc"]["links"]
    ):
        return False
    return True


def _section(title: str, value: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _render_scene_fact_video(anchor_id: str, data_root: Path) -> None:
    command = [
        sys.executable,
        "-u",
        "-m",
        "step7.review_scene_fact",
        "--anchor-id",
        anchor_id,
        "--dataset-root",
        str(data_root),
        "--force",
    ]
    print("\n[Step 8 review] Rendering Step 7 scene-review video", flush=True)
    print("[Step 8 review] Command: " + " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(
            "Step 7 video renderer failed. The Structured CoC was not presented "
            "without its visual evidence."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--anchor-id")
    parser.add_argument("--rule-id")
    parser.add_argument("--quality", choices=("usable", "partial", "unknown"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Print the structured review only. Video is enabled by default.",
    )
    args = parser.parse_args()

    paths = Step8Paths.from_data_root(args.data_root)
    print("[Step 8 review] Validating frozen inputs", flush=True)
    joined_rows, _ = load_and_validate_inputs(paths)

    matches: list[tuple[JoinedAnchor, dict[str, Any]]] = []
    for joined in joined_rows:
        record = build_structured_causality(joined)
        if _matches(
            joined,
            record,
            anchor_id=args.anchor_id,
            rule_id=args.rule_id,
            quality=args.quality,
        ):
            matches.append((joined, record))

    if not matches:
        raise SystemExit("No Step 8 record matched the requested filters")

    selected_joined, selected_record = random.Random(args.seed).choice(matches)
    print(f"[Step 8 review] Matched records: {len(matches)}")
    print(f"[Step 8 review] Selected anchor: {selected_joined.anchor_id}")

    if not args.no_video:
        _render_scene_fact_video(selected_joined.anchor_id, paths.data_root)

    navigation = selected_joined.navigation.get("navigation", {})
    scene = selected_joined.scene_fact
    meta = selected_joined.meta_action
    _section("Identity", {
        "anchor_id": selected_joined.anchor_id,
        "clip_id": selected_joined.clip_id,
        "anchor_ns": selected_joined.anchor_ns,
    })
    _section("Navigation", navigation)
    _section("Road context", scene.get("road_context", {}))
    _section("Lead actors", scene.get("actor_context", {}).get("lead_actors", []))
    _section("Left nearby actors", scene.get("actor_context", {}).get("left_nearby_actors", []))
    _section("Right nearby actors", scene.get("actor_context", {}).get("right_nearby_actors", []))
    _section("Meta-action", {
        "longitudinal": meta.get("longitudinal", {}),
        "lateral": meta.get("lateral", {}),
        "overall_quality_status": meta.get("overall_quality_status"),
    })
    _section("Structured CoC", selected_record["structured_coc"])
    _section("Step 8 quality", selected_record["quality"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
