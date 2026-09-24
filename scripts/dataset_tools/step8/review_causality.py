#!/usr/bin/env python3
"""Review one formal Step 8 Structured CoC record with synchronized video."""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .causality import build_structured_causality
from .contract import JoinedAnchor, Step8Paths, load_and_validate_inputs

FORMAL_OUTPUT_NAME = "structured_causality.jsonl"


def read_formal_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"formal Step 8 output not found: {path}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            anchor_id = str(row.get("anchor_id", ""))
            if not anchor_id:
                raise ValueError(f"{path}:{line_number}: missing anchor_id")
            if anchor_id in rows:
                raise ValueError(f"duplicate formal Step 8 Anchor: {anchor_id}")
            rows[anchor_id] = row
    return rows


def _matches(
    joined: JoinedAnchor,
    record: Mapping[str, Any],
    *,
    anchor_id: str | None,
    rule_id: str | None,
    quality: str | None,
    relation: str | None,
    actor_class: str | None,
) -> bool:
    if anchor_id is not None and joined.anchor_id != anchor_id:
        return False
    if quality is not None and record["quality"]["status"] != quality:
        return False
    links = record["structured_coc"]["links"]
    if rule_id is not None and not any(link["rule_id"] == rule_id for link in links):
        return False
    if relation is not None and not any(link["relation"] == relation for link in links):
        return False
    if actor_class is not None and not any(
        node["node_type"] == "actor_state"
        and node.get("value", {}).get("actor_class") == actor_class
        for node in record["structured_coc"]["nodes"]
    ):
        return False
    return True


def _section(title: str, value: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _render_scene_fact_video(anchor_id: str, data_root: Path) -> None:
    command = [
        sys.executable, "-u", "-m", "step7.review_scene_fact",
        "--anchor-id", anchor_id,
        "--dataset-root", str(data_root),
        "--force",
    ]
    print("\n[Step 8 review] Rendering Step 7 scene-review video", flush=True)
    print("[Step 8 review] Command: " + " ".join(command), flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(
            "[Step 8 review] WARNING: video renderer failed; continuing with "
            "formal Structured CoC diagnostics.",
            file=sys.stderr,
            flush=True,
        )


def _participating_actor_rows(
    joined: JoinedAnchor,
    record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    context = joined.scene_fact.get("actor_context", {})
    actor_by_identity = {}
    for list_name, role in (
        ("lead_actors", "lead"),
        ("left_nearby_actors", "left"),
        ("right_nearby_actors", "right"),
    ):
        for actor in context.get(list_name, []):
            actor_by_identity[(role, actor.get("role_rank"), str(actor.get("track_id")))] = actor
    selected = []
    for node in record["structured_coc"]["nodes"]:
        if node["node_type"] != "actor_state":
            continue
        ref = node.get("value", {}).get("actor_ref", {})
        key = (ref.get("scene_role"), ref.get("role_rank"), str(ref.get("track_id")))
        selected.append({
            "node_id": node["node_id"],
            "scene_fact_actor": actor_by_identity.get(key),
            "coc_value": node.get("value", {}),
        })
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--formal-output", type=Path, default=None)
    parser.add_argument("--anchor-id")
    parser.add_argument("--rule-id")
    parser.add_argument("--quality", choices=("usable", "partial", "unknown"))
    parser.add_argument("--relation", choices=("aligns_with", "supports", "insufficient_evidence"))
    parser.add_argument("--actor-class")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--show-full-scene-fact", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = Step8Paths.from_data_root(args.data_root)
    formal_output = args.formal_output or (
        paths.data_root / "annotations" / "v0.1-draft" / FORMAL_OUTPUT_NAME
    )

    print("[Step 8 review] Validating frozen inputs", flush=True)
    joined_rows, _ = load_and_validate_inputs(paths)
    formal_records = read_formal_records(formal_output)
    if set(formal_records) != {row.anchor_id for row in joined_rows}:
        raise ValueError("formal Step 8 Anchors do not close to frozen joined inputs")

    matches: list[tuple[JoinedAnchor, dict[str, Any]]] = []
    for joined in joined_rows:
        formal = formal_records[joined.anchor_id]
        regenerated = build_structured_causality(joined)
        if regenerated != formal:
            raise ValueError(
                f"formal Step 8 record differs from current rules: {joined.anchor_id}"
            )
        if _matches(
            joined,
            formal,
            anchor_id=args.anchor_id,
            rule_id=args.rule_id,
            quality=args.quality,
            relation=args.relation,
            actor_class=args.actor_class,
        ):
            matches.append((joined, formal))

    if not matches:
        raise SystemExit("No formal Step 8 record matched the requested filters")

    selected_joined, selected_record = random.Random(args.seed).choice(matches)
    print(f"[Step 8 review] Matched records: {len(matches)}")
    print(f"[Step 8 review] Selected anchor: {selected_joined.anchor_id}")
    print("[Step 8 review] Formal/runtime consistency: PASS")

    if not args.no_video:
        _render_scene_fact_video(selected_joined.anchor_id, paths.data_root)

    _section("Identity", {
        "anchor_id": selected_joined.anchor_id,
        "clip_id": selected_joined.clip_id,
        "anchor_ns": selected_joined.anchor_ns,
    })
    _section("Navigation", selected_joined.navigation.get("navigation", {}))
    _section("Meta-action", {
        "longitudinal": selected_joined.meta_action.get("longitudinal", {}),
        "lateral": selected_joined.meta_action.get("lateral", {}),
        "overall_quality_status": selected_joined.meta_action.get("overall_quality_status"),
    })
    _section("Participating Scene-Fact Actors", _participating_actor_rows(selected_joined, selected_record))
    _section("Formal Structured CoC", selected_record["structured_coc"])
    _section("Step 8 quality", selected_record["quality"])
    if args.show_full_scene_fact:
        _section("Full Scene Fact", selected_joined.scene_fact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
