#!/usr/bin/env python3
"""Unified production entry point for Dataset Step 6."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def step6_commands(*, force: bool) -> tuple[tuple[str, ...], ...]:
    scripts = (
        "profile_navigation_branch_context_v01.py",
        "profile_road_level_navigation_features_v01.py",
        "profile_navigation_route_features_v01.py",
        "generate_navigation_candidates_v01.py",
        "finalize_navigation_v01.py",
    )
    commands = []
    for script in scripts:
        command = [sys.executable, script]
        if force:
            command.append("--force")
        commands.append(tuple(command))
    return tuple(commands)


def run_step6_pipeline(*, force: bool) -> None:
    commands = step6_commands(force=force)
    for index, command in enumerate(commands, start=1):
        print()
        print("=" * 78)
        print(f"STEP 6 STAGE {index}/{len(commands)}")
        print("Command:", " ".join(command))
        print("=" * 78)
        completed = subprocess.run(command, cwd=SCRIPT_DIR, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "Step 6 stage failed with exit code "
                f"{completed.returncode}: {' '.join(command)}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Pass --force to all five Step 6 stages.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_step6_pipeline(force=args.force)
    print()
    print("PASS: Step 6 pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
