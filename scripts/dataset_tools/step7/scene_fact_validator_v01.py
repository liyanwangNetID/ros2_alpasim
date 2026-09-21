"""JSON-Schema validation for Step 7L final Scene Facts."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from jsonschema import Draft202012Validator


class SceneFactValidationError(ValueError):
    pass


def load_scene_fact_validator(schema_path: Path) -> Draft202012Validator:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_scene_fact_record(
    record: Mapping[str, Any], *, validator: Draft202012Validator
) -> None:
    errors = sorted(validator.iter_errors(record), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        path = ".".join(str(value) for value in error.absolute_path) or "$"
        raise SceneFactValidationError(f"{path}: {error.message}")
