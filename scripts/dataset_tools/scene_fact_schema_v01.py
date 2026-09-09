#!/usr/bin/env python3
"""Step 7 v0.1 Scene-Fact schema constants.

This module defines the stable vocabulary shared by Step 7 feature
profiling, rule generation, validation, summaries, and tests.

It contains no dataset access or classification thresholds.
"""

from __future__ import annotations


SCENE_FACT_FORMAT_VERSION = "0.1-draft"
OBSERVABILITY_FORMAT_VERSION = "0.1-draft"
FEATURE_FORMAT_VERSION = "0.1-draft"
GENERATOR_VERSION = "0.1.0"
RULE_VERSION = "scene_fact_rules_v0.1-draft"


CAMERA_NAMES = (
    "front_wide",
    "front_tele",
    "cross_left",
    "cross_right",
)


ROAD_CONTEXT_TYPES = frozenset(
    {
        "lane_following",
        "intersection_approach",
        "intersection",
        "unknown",
    }
)


PROXIMITY_STATUSES = frozenset(
    {
        "at",
        "near",
        "approaching",
        "far",
        "none",
        "unknown",
    }
)


FINAL_PRESENCE_STATUSES = frozenset(
    {
        "present",
        "not_present",
        "unknown",
    }
)


OBSERVABILITY_STATUSES = frozenset(
    {
        "candidate_visible",
        "partially_occluded",
        "heavily_occluded",
        "not_visible",
        "unknown",
    }
)


INTERNAL_VISIBILITY_DECISIONS = frozenset(
    {
        "included",
        "not_observed",
        "rejected",
        "unknown",
    }
)


RELATIVE_POSITION_REGIONS = frozenset(
    {
        "front",
        "front_left",
        "front_right",
        "left",
        "right",
        "rear_left",
        "rear",
        "rear_right",
        "overlapping",
        "unknown",
    }
)


RELATIVE_DISTANCE_CATEGORIES = frozenset(
    {
        "near",
        "medium",
        "far",
        "unknown",
    }
)


DISTANCE_TREND_CATEGORIES = frozenset(
    {
        "approaching",
        "receding",
        "stable_distance",
        "uncertain",
    }
)


RELATIVE_SPEED_CATEGORIES = frozenset(
    {
        "slower_than_ego",
        "similar_to_ego",
        "faster_than_ego",
        "stationary",
        "uncertain",
    }
)


QUALITY_STATUSES = frozenset(
    {
        "usable",
        "unknown",
    }
)


ACTOR_ROLE_KEYS = (
    "lead_vehicle",
    "left_nearby_vehicle",
    "right_nearby_vehicle",
)


FINAL_RECORD_REQUIRED_KEYS = frozenset(
    {
        "scene_fact_format_version",
        "generator_version",
        "rule_version",
        "anchor_id",
        "clip_id",
        "anchor_ns",
        "road_context",
        "lead_vehicle",
        "left_nearby_vehicle",
        "right_nearby_vehicle",
        "quality",
    }
)


FORBIDDEN_FUTURE_INPUTS = (
    "actors/future.jsonl",
    "ego/ground_truth_future.jsonl",
    "ego/complete_recording_ground_truth.json",
    "ego/planner_output.jsonl",
)


ALLOWED_CURRENT_OR_PAST_INPUTS = (
    "keyframes.jsonl",
    "ego/ego_state.jsonl",
    "actors/current.jsonl",
    "calibration/*.json",
    "cameras/*/timestamps.jsonl",
    "map/vector_map.json",
)
