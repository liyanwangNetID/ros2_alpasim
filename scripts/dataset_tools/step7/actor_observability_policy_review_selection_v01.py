"""Deterministically select Step 7E policy comparison review cases.

The selector consumes the selected-policy impact report and creates a compact,
Clip-diverse manifest. It does not read images, modify geometric evidence, or
write final observability labels.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

WINNING_ONLY = "rejected_by_winning_cell_candidate_only"
FRACTION_ONLY = "rejected_by_visible_fraction_candidate_only"
BOTH = "rejected_by_both_candidates"
SUPPORTED_CATEGORIES = (WINNING_ONLY, FRACTION_ONLY, BOTH)


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["anchor_id"]), str(row["track_id"])


def _validate_row(row: Mapping[str, Any], index: int) -> None:
    required = (
        "anchor_id",
        "clip_id",
        "track_id",
        "label_class",
        "maximum_visible_fraction",
        "total_winning_cell_count",
        "change_category",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise ValueError(f"affected row {index} is missing fields: {missing}")
    if row["change_category"] not in SUPPORTED_CATEGORIES:
        raise ValueError(
            f"affected row {index} has unsupported change_category: "
            f"{row['change_category']}"
        )


def _review_priority(row: Mapping[str, Any]) -> tuple[Any, ...]:
    category = str(row["change_category"])
    fraction = float(row["maximum_visible_fraction"])
    winning = int(row["total_winning_cell_count"])
    stable_identity = _identity(row)
    if category == WINNING_ONLY:
        # Test whether a one-cell rejection would remove otherwise strong cases.
        return (-fraction, stable_identity)
    if category == FRACTION_ONLY:
        # Test whether a fraction floor would remove cases with substantial
        # absolute winning support, then inspect the smallest fractions first.
        return (-winning, fraction, stable_identity)
    # Cases near the 0.01 boundary are most informative for the disagreement.
    return (-fraction, stable_identity)


def _select_clip_diverse(
    rows: Sequence[Mapping[str, Any]], *, quota: int
) -> tuple[Mapping[str, Any], ...]:
    ordered = sorted(rows, key=_review_priority)
    selected: list[Mapping[str, Any]] = []
    selected_identities: set[tuple[str, str]] = set()
    selected_clips: set[str] = set()

    for row in ordered:
        clip_id = str(row["clip_id"])
        if clip_id in selected_clips:
            continue
        selected.append(row)
        selected_identities.add(_identity(row))
        selected_clips.add(clip_id)
        if len(selected) == quota:
            return tuple(selected)

    for row in ordered:
        identity = _identity(row)
        if identity in selected_identities:
            continue
        selected.append(row)
        selected_identities.add(identity)
        if len(selected) == quota:
            break
    return tuple(selected)


def select_policy_review_cases(
    *, impact_report: Mapping[str, Any], quota_per_category: int = 12
) -> dict[str, Any]:
    """Build a deterministic, category-balanced visual-review manifest."""
    if quota_per_category <= 0:
        raise ValueError("quota_per_category must be positive")
    schema = impact_report.get("schema_version")
    if schema != "step7e-observability-selected-policy-impact-v01":
        raise ValueError(f"unexpected impact report schema: {schema}")

    source_rows = tuple(impact_report.get("affected_rows", ()))
    seen: set[tuple[str, str]] = set()
    grouped: dict[str, list[Mapping[str, Any]]] = {
        category: [] for category in SUPPORTED_CATEGORIES
    }
    for index, row in enumerate(source_rows):
        _validate_row(row, index)
        identity = _identity(row)
        if identity in seen:
            raise ValueError("affected Anchor/Actor identities must be unique")
        seen.add(identity)
        grouped[str(row["change_category"])].append(row)

    cases: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for category in SUPPORTED_CATEGORIES:
        selected = _select_clip_diverse(
            grouped[category], quota=min(quota_per_category, len(grouped[category]))
        )
        for rank, row in enumerate(selected, start=1):
            case = dict(row)
            case["review_category"] = category
            case["review_rank_within_category"] = rank
            case["review_reason"] = {
                WINNING_ONLY: (
                    "One winning cell but visible fraction remains at or above "
                    "the 0.01 comparison floor; inspect whether the cell is "
                    "valid object evidence or isolated raster noise."
                ),
                FRACTION_ONLY: (
                    "At least two winning cells but visible fraction is below "
                    "0.01; inspect whether the low fraction reflects genuine "
                    "occlusion or scale/raster behavior."
                ),
                BOTH: (
                    "Exactly one winning cell and visible fraction below 0.01; "
                    "inspect a boundary case rejected by both candidates."
                ),
            }[category]
            cases.append(case)
            selected_counts[category] += 1

    selected_identities = [_identity(case) for case in cases]
    if len(selected_identities) != len(set(selected_identities)):
        raise RuntimeError("review case identities are not unique")

    return {
        "schema_version": "step7e-observability-policy-review-selection-v01",
        "description": (
            "Deterministic Clip-diverse review selection for the two narrowed "
            "dynamic-occlusion candidates. This is a review manifest only."
        ),
        "source_schema_version": schema,
        "quota_per_category": quota_per_category,
        "source_affected_actor_count": len(source_rows),
        "source_category_counts": {
            key: len(grouped[key]) for key in SUPPORTED_CATEGORIES
        },
        "selected_case_count": len(cases),
        "selected_clip_count": len({str(case["clip_id"]) for case in cases}),
        "selected_category_counts": {
            key: selected_counts[key] for key in SUPPORTED_CATEGORIES
        },
        "selection_order": list(SUPPORTED_CATEGORIES),
        "cases": cases,
    }
