from step1.clip_manifest import CAMERA_NAMES, MANIFEST_VERSION, SCRIPT_VERSION
from step1.clip_schema import build_clip_schema_markdown, write_clip_schema


def test_schema_uses_implemented_versions_and_camera_order():
    text = build_clip_schema_markdown()
    assert f"Manifest schema version: `{MANIFEST_VERSION}`" in text
    assert f"Manifest builder version: `{SCRIPT_VERSION}`" in text
    positions = [text.index(f"- `{camera}`") for camera in CAMERA_NAMES]
    assert positions == sorted(positions)


def test_schema_freezes_paths_time_and_quality_contract():
    text = build_clip_schema_markdown()
    for value in (
        "manifests/clips_v0.1.jsonl",
        "reports/clip_manifest_summary_v0.1.json",
        "integer nanoseconds",
        "metadata_readable",
        "validation_readable",
        "required_files_present",
        "camera_indexes_readable",
        "camera_images_complete",
        "camera_timestamps_valid",
        "complete_ground_truth_present",
        "vector_map_present",
        "validation_valid",
    ):
        assert value in text


def test_write_is_deterministic_and_requires_force(tmp_path):
    output = tmp_path / "clip_schema.md"
    write_clip_schema(output)
    first = output.read_bytes()
    try:
        write_clip_schema(output)
    except FileExistsError:
        pass
    else:
        raise AssertionError("unforced overwrite should fail")
    write_clip_schema(output, force=True)
    assert output.read_bytes() == first


def test_default_output_uses_configured_schema_root():
    from project_paths import SCHEMA_ROOT
    from step1.clip_schema import DEFAULT_OUTPUT
    assert DEFAULT_OUTPUT == SCHEMA_ROOT / "clip_schema.md"
