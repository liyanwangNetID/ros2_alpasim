#!/usr/bin/env python3
"""Inspect winning-presence changes in the multicase occlusion profile.

Reads the existing threshold-free JSON report and prints every Actor whose
winning-cell presence changes between 480x270 and 640x360. No geometry or
Z-buffer computation is repeated and no acceptance threshold is applied.
"""

from __future__ import annotations

import json
from pathlib import Path


INPUT = Path("/tmp/step7e_multicase_occlusion_resolution_profile.json")


def main() -> int:
    report = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = []

    for case in report["cases"]:
        anchor_id = str(case["anchor_id"])
        camera_name = str(case["camera_name"])
        for comparison in case["comparisons"]:
            if comparison["winning_presence_changed"] is True:
                rows.append((anchor_id, camera_name, comparison))

    expected_count = int(
        report["aggregate"]["winning_presence_change_count"]
    )
    if len(rows) != expected_count:
        raise RuntimeError(
            "Winning-presence row count does not match aggregate: "
            f"rows={len(rows)}, aggregate={expected_count}"
        )

    print("Winning-presence changes")
    print("input:", INPUT)
    print("count:", len(rows))

    for anchor_id, camera_name, row in rows:
        print()
        print("=" * 100)
        print(anchor_id, camera_name, f"track={row['track_id']}")
        print("=" * 100)
        print(
            "candidate_raster:",
            f"{row['candidate_raster_width']}x"
            f"{row['candidate_raster_height']}",
        )
        print(
            "reference_raster:",
            f"{row['reference_raster_width']}x"
            f"{row['reference_raster_height']}",
        )
        print(
            "candidate:",
            f"status={row['candidate_evidence_status']}",
            f"occupied={row['candidate_occupied_cell_count']}",
            f"winning={row['candidate_winning_cell_count']}",
            f"visible_fraction={row['candidate_visible_fraction']}",
            f"occluders={row['candidate_occluding_actor_ids']}",
        )
        print(
            "reference:",
            f"status={row['reference_evidence_status']}",
            f"occupied={row['reference_occupied_cell_count']}",
            f"winning={row['reference_winning_cell_count']}",
            f"visible_fraction={row['reference_visible_fraction']}",
            f"occluders={row['reference_occluding_actor_ids']}",
        )
        print(
            "changes:",
            f"surface_presence={row['sampled_surface_presence_changed']}",
            f"winning_presence={row['winning_presence_changed']}",
            f"occluder_set={row['occluding_actor_set_changed']}",
            f"absolute_delta={row['absolute_visible_fraction_delta']}",
        )

    print()
    print(
        "PASS: all multicase winning-presence changes displayed from the "
        "threshold-free profile."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
