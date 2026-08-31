#!/usr/bin/env python3
"""Unified production entry point for Dataset Step 5.

Runs event detection, semantic event deduplication, and final keyframe
selection in the frozen order. Existing stage scripts remain the source of
truth for stage-specific logic and output schemas.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def step5_commands(*, force: bool, reuse_existing_events: bool) -> tuple[tuple[str, ...], ...]:
    commands: list[tuple[str, ...]] = []

    if not reuse_existing_events:
        detect = [sys.executable, "detect_keyframe_events_v01.py"]
        deduplicate = [sys.executable, "deduplicate_keyframe_events_v01.py"]
        if force:
            detect.append("--force")
            deduplicate.append("--force")
        commands.extend((tuple(detect), tuple(deduplicate)))

    select = [sys.executable, "select_keyframes_v01.py"]
    if force:
        select.append("--force")
    commands.append(tuple(select))

    return tuple(commands)


def run_step5_pipeline(*, force: bool, reuse_existing_events: bool) -> None:
    commands = step5_commands(
        force=force,
        reuse_existing_events=reuse_existing_events,
    )

    for index, command in enumerate(commands, start=1):
        print()
        print("=" * 78)
        print(f"STEP 5 STAGE {index}/{len(commands)}")
        print("Command:", " ".join(command))
        print("=" * 78)

        completed = subprocess.run(
            command,
            cwd=SCRIPT_DIR,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Step 5 stage failed with exit code "
                f"{completed.returncode}: {' '.join(command)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to every executed Step 5 stage.",
    )
    parser.add_argument(
        "--reuse-existing-events",
        action="store_true",
        help=(
            "Skip event detection and deduplication, and rebuild final "
            "Keyframes from existing deduplicated event candidates."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_step5_pipeline(
        force=args.force,
        reuse_existing_events=args.reuse_existing_events,
    )
    print()
    print("PASS: Step 5 pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
