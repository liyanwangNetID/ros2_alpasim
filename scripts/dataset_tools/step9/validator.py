from __future__ import annotations

import re
from typing import Any, Mapping

from .contract import Step9ContractError, response_schema, validate_schema

UNCERTAINTY_TERMS = (
    "unknown",
    "uncertain",
    "insufficient",
    "limited",
    "limitation",
    "unavailable",
    "may describe",
    "cannot determine",
    "cannot be determined",
    "unable to determine",
    "not enough evidence",
    "lack of evidence",
    "lacks evidence",
    "no supported causal link",
)
FORBIDDEN_TERMS = (
    "link_",
    "node_",
    "rule_",
    "track_id",
    "sha256",
    ".jsonl",
    "provenance",
)
LONGITUDINAL_ACTION_PATTERNS = (
    r"\baccelerat(?:e|es|ed|ing|ion)\b",
    r"\bmaintain(?:s|ed|ing)? speed\b",
    r"\bdecelerat(?:e|es|ed|ing|ion)\b",
    r"\bslow(?:s|ed|ing)? down\b",
    r"\bstop(?:s|ped|ping)?\b",
    r"\blongitudinal\b",
)
SUPPORT_LANGUAGE = (
    "supported",
    "supports",
    "consistent with",
    "justified",
    "because",
    "due to",
    "therefore",
)
QUALITY_AS_REASON_PATTERNS = (
    r"consistent with (?:the )?(?:overall )?(?:usable|partial|unknown) quality",
    r"(?:because|due to) (?:the )?(?:overall )?(?:usable|partial|unknown) quality",
    r"quality status (?:supports|justifies)",
)


def _supports_target(package: Mapping[str, Any], target_type: str) -> bool:
    return any(
        item.get("relation") == "supports"
        and (item.get("facts") or {}).get("target_type") == target_type
        for item in package["evidence"]
    )


def _mentions_unsupported_longitudinal_claim(summary: str) -> bool:
    lowered = summary.lower().replace("_", " ")
    has_action = any(
        re.search(pattern, lowered)
        for pattern in LONGITUDINAL_ACTION_PATTERNS
    )
    has_support = any(term in lowered for term in SUPPORT_LANGUAGE)
    return has_action and has_support


def validate_response(
    response: Mapping[str, Any],
    package: Mapping[str, Any],
) -> dict[str, Any]:
    validate_schema(response, response_schema())
    available = {item["evidence_key"] for item in package["evidence"]}
    used = response["used_evidence_keys"]
    if not set(used) <= available:
        raise Step9ContractError(
            f"unknown evidence keys: {sorted(set(used) - available)}"
        )

    summary = response["reasoning_summary"]
    lowered = summary.lower()
    forbidden = [term for term in FORBIDDEN_TERMS if term in lowered]
    if forbidden:
        raise Step9ContractError(f"forbidden provenance terms: {forbidden}")

    if any(re.search(pattern, lowered) for pattern in QUALITY_AS_REASON_PATTERNS):
        raise Step9ContractError("quality status must not be used as an action reason")

    has_longitudinal_support = _supports_target(
        package,
        "longitudinal_decision",
    )
    if not has_longitudinal_support and _mentions_unsupported_longitudinal_claim(summary):
        raise Step9ContractError(
            "summary claims support for a longitudinal action without longitudinal evidence"
        )

    quality = package["quality_status"]
    limitations = response["limitations"]
    if quality in {"partial", "unknown"}:
        if not limitations:
            raise Step9ContractError(f"{quality} response requires limitations")
        if not any(term in lowered for term in UNCERTAINTY_TERMS):
            raise Step9ContractError(
                f"{quality} summary must state uncertainty"
            )
    return dict(response)
