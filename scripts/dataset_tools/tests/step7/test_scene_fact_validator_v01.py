import json
from pathlib import Path
import pytest
from step7.scene_fact_validator_v01 import (
    SceneFactValidationError, load_scene_fact_validator, validate_scene_fact_record,
)


def test_validator_accepts_minimal_valid_record(tmp_path: Path):
    schema={"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object",
            "additionalProperties":False,"required":["x"],"properties":{"x":{"type":"integer"}}}
    path=tmp_path/"schema.json"; path.write_text(json.dumps(schema))
    validator=load_scene_fact_validator(path)
    validate_scene_fact_record({"x":1},validator=validator)
    with pytest.raises(SceneFactValidationError):
        validate_scene_fact_record({"x":"bad"},validator=validator)
