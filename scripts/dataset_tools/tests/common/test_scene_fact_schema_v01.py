#!/usr/bin/env python3
"""Tests for the Step 7 v0.1 Scene-Fact vocabulary."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path




from project_paths import SCHEMA_ROOT  # noqa: E402
from scene_fact_schema_v01 import (  # noqa: E402
    ACTOR_ROLE_KEYS,
    CAMERA_NAMES,
    DISTANCE_TREND_CATEGORIES,
    FINAL_PRESENCE_STATUSES,
    FINAL_RECORD_REQUIRED_KEYS,
    FORBIDDEN_FUTURE_INPUTS,
    OBSERVABILITY_STATUSES,
    PROXIMITY_STATUSES,
    QUALITY_STATUSES,
    RELATIVE_DISTANCE_CATEGORIES,
    RELATIVE_POSITION_REGIONS,
    RELATIVE_SPEED_CATEGORIES,
    ROAD_CONTEXT_TYPES,
    SCENE_FACT_FORMAT_VERSION,
)


class SceneFactSchemaTests(unittest.TestCase):
    def test_identity_and_role_fields_are_required(self):
        for key in (
            "anchor_id",
            "clip_id",
            "anchor_ns",
            *ACTOR_ROLE_KEYS,
        ):
            self.assertIn(key, FINAL_RECORD_REQUIRED_KEYS)

    def test_final_presence_does_not_reveal_hidden_actor(self):
        self.assertNotIn(
            "not_observed",
            FINAL_PRESENCE_STATUSES,
        )
        self.assertEqual(
            FINAL_PRESENCE_STATUSES,
            {"present", "not_present", "unknown"},
        )

    def test_first_version_has_four_selected_cameras(self):
        self.assertEqual(
            CAMERA_NAMES,
            (
                "front_wide",
                "front_tele",
                "cross_left",
                "cross_right",
            ),
        )

    def test_conservative_unknown_states_exist(self):
        self.assertIn("unknown", ROAD_CONTEXT_TYPES)
        self.assertIn("unknown", OBSERVABILITY_STATUSES)
        self.assertIn("unknown", QUALITY_STATUSES)
        self.assertIn("uncertain", DISTANCE_TREND_CATEGORIES)
        self.assertIn("uncertain", RELATIVE_SPEED_CATEGORIES)

    def test_future_sources_are_explicitly_forbidden(self):
        self.assertIn(
            "actors/future.jsonl",
            FORBIDDEN_FUTURE_INPUTS,
        )
        self.assertIn(
            "ego/ground_truth_future.jsonl",
            FORBIDDEN_FUTURE_INPUTS,
        )
        self.assertIn(
            "ego/planner_output.jsonl",
            FORBIDDEN_FUTURE_INPUTS,
        )

    def test_schema_version_is_draft(self):
        self.assertEqual(
            SCENE_FACT_FORMAT_VERSION,
            "0.1-draft",
        )

    def test_json_schema_uses_configured_schema_root(self):
        schema_path = (
            SCHEMA_ROOT
            / "scene_fact_schema_v0.1-draft.json"
        )

        self.assertTrue(schema_path.is_file())

    def test_python_vocabulary_matches_json_schema(self):
        schema_path = (
            SCHEMA_ROOT
            / "scene_fact_schema_v0.1-draft.json"
        )
        schema = json.loads(
            schema_path.read_text(encoding="utf-8")
        )
        definitions = schema["$defs"]

        comparisons = {
            "cameraName": set(CAMERA_NAMES),
            "proximityStatus": set(PROXIMITY_STATUSES),
            "relativePosition": set(
                RELATIVE_POSITION_REGIONS
            ),
            "relativeDistance": set(
                RELATIVE_DISTANCE_CATEGORIES
            ),
            "distanceTrend": set(
                DISTANCE_TREND_CATEGORIES
            ),
            "relativeSpeed": set(
                RELATIVE_SPEED_CATEGORIES
            ),
            "qualityStatus": set(QUALITY_STATUSES),
        }

        for definition_name, expected in comparisons.items():
            with self.subTest(
                definition_name=definition_name
            ):
                self.assertEqual(
                    set(
                        definitions[definition_name]["enum"]
                    ),
                    expected,
                )

        self.assertEqual(
            set(
                definitions["roadContext"]
                ["properties"]["type"]["enum"]
            ),
            set(ROAD_CONTEXT_TYPES),
        )

        self.assertEqual(
            set(
                definitions["absentActorRole"]
                ["properties"]["presence_status"]["enum"]
            ),
            FINAL_PRESENCE_STATUSES
            - {"present"},
        )

        self.assertEqual(
            set(
                definitions["presentActorRole"]
                ["properties"]
                ["observability_status"]["enum"]
            ),
            {
                "candidate_visible",
                "partially_occluded",
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
