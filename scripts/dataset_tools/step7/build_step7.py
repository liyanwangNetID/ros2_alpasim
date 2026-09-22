#!/usr/bin/env python3
"""Unified Step 7 build entrypoint.

Runs the current production stages in dependency order. Child stages retain
their own progress, rate, and ETA output. The unified entrypoint adds stage
boundaries, elapsed time, failure propagation, and selective range execution.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, slots=True)
class BuildStage:
    name: str
    module: str
    description: str


STAGES: tuple[BuildStage, ...] = (
    BuildStage(
        "projection_evidence",
        "step7.build_projection_evidence",
        "Build projection evidence and Actor observability inputs.",
    ),
    BuildStage(
        "occlusion",
        "step7.build_occlusion",
        "Build formal geometric occlusion evidence.",
    ),
    BuildStage(
        "observability_policy",
        "step7.build_observability",
        "Build the frozen dynamic-occlusion policy report.",
    ),
    BuildStage(
        "history",
        "step7.build_history",
        "Build Actor short-history features.",
    ),
    BuildStage(
        "road_context",
        "step7.build_road_context",
        "Build Ego and Actor road-context features.",
    ),
    BuildStage(
        "actor_roles",
        "step7.build_actor_roles",
        "Build deterministic Actor-role selections.",
    ),
    BuildStage(
        "scene_features",
        "step7.build_scene_features",
        "Build the unified Scene-Fact feature layer.",
    ),
    BuildStage(
        "scene_facts",
        "step7.build_scene_facts",
        "Build and validate final Scene Facts.",
    ),
)


def duration_text(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def stage_index(name: str) -> int:
    for index, stage in enumerate(STAGES):
        if stage.name == name:
            return index
    raise ValueError(f"unknown Step 7 stage: {name}")


def select_stages(
    *,
    from_stage: str | None,
    to_stage: str | None,
    skipped: Sequence[str],
) -> tuple[BuildStage, ...]:
    start = 0 if from_stage is None else stage_index(from_stage)
    end = len(STAGES) - 1 if to_stage is None else stage_index(to_stage)
    if start > end:
        raise ValueError("from-stage must not occur after to-stage")
    unknown = sorted(set(skipped) - {stage.name for stage in STAGES})
    if unknown:
        raise ValueError(f"unknown skipped stages: {unknown}")
    skipped_set = set(skipped)
    return tuple(
        stage
        for stage in STAGES[start : end + 1]
        if stage.name not in skipped_set
    )


def print_stage_list() -> None:
    for index, stage in enumerate(STAGES, start=1):
        print(
            f"{index}. {stage.name}: {stage.module}\n"
            f"   {stage.description}"
        )


def run_stage(
    stage: BuildStage,
    *,
    position: int,
    total: int,
    dry_run: bool,
) -> float:
    command = (sys.executable, "-u", "-m", stage.module)
    print(
        f"[Step 7] stage {position}/{total}: {stage.name}\n"
        f"[Step 7] module: {stage.module}\n"
        f"[Step 7] description: {stage.description}",
        flush=True,
    )
    if dry_run:
        print(
            "[Step 7] dry-run command: " + " ".join(command),
            flush=True,
        )
        return 0.0
    started = time.monotonic()
    completed = subprocess.run(command, check=False)
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"Step 7 stage {stage.name} failed with exit code "
            f"{completed.returncode} after {duration_text(elapsed)}"
        )
    print(
        f"[Step 7] completed {stage.name} in {duration_text(elapsed)}",
        flush=True,
    )
    return elapsed


def build_parser() -> argparse.ArgumentParser:
    names = tuple(stage.name for stage in STAGES)
    parser = argparse.ArgumentParser(
        description="Build Step 7 products in dependency order.",
    )
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List the ordered stages and exit.",
    )
    parser.add_argument(
        "--from-stage",
        choices=names,
        help="Start from this stage, inclusive.",
    )
    parser.add_argument(
        "--to-stage",
        choices=names,
        help="Stop after this stage, inclusive.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=names,
        metavar="STAGE",
        help="Skip one named stage. May be specified repeatedly.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected commands without running them.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.list_stages:
        print_stage_list()
        return 0
    try:
        selected = select_stages(
            from_stage=arguments.from_stage,
            to_stage=arguments.to_stage,
            skipped=arguments.skip,
        )
    except ValueError as error:
        parser.error(str(error))
    if not selected:
        parser.error("no Step 7 stages were selected")
    started = time.monotonic()
    durations: list[tuple[str, float]] = []
    print(
        f"[Step 7] selected stages: {len(selected)}",
        flush=True,
    )
    for position, stage in enumerate(selected, start=1):
        elapsed = run_stage(
            stage,
            position=position,
            total=len(selected),
            dry_run=arguments.dry_run,
        )
        durations.append((stage.name, elapsed))
    total_elapsed = time.monotonic() - started
    print("[Step 7] stage summary:", flush=True)
    for name, elapsed in durations:
        print(
            f"[Step 7] {name}: {duration_text(elapsed)}",
            flush=True,
        )
    print(
        f"[Step 7] total elapsed: {duration_text(total_elapsed)}",
        flush=True,
    )
    print("PASS: unified Step 7 build completed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
