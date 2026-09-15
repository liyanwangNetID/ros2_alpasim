from types import SimpleNamespace

from step7 import export_step7e_geometric_occlusion_evidence_v02 as target


def test_build_anchor_uses_enhanced_pipeline_and_prepares_inputs(monkeypatch):
    calls = []
    keyframe = {"anchor_id": "clip_100", "clip_id": "clip", "anchor_ns": 100}
    ego = SimpleNamespace(message={"ego": True})
    actors = ({"track_id": "1"},)
    cameras = {"camera": object()}
    enhanced = SimpleNamespace(actor_count=1)

    monkeypatch.setattr(target, "exact_inputs", lambda reader, ns: (ego, actors))
    monkeypatch.setattr(
        target,
        "load_camera_inputs",
        lambda reader, anchor_id, ns: cameras,
    )

    def build(**kwargs):
        calls.append(("build", kwargs))
        return enhanced

    def prepare(**kwargs):
        calls.append(("prepare", kwargs))
        return ("prepared",)

    monkeypatch.setattr(
        target,
        "build_actor_geometric_occlusion_with_projection_context",
        build,
    )
    monkeypatch.setattr(
        target,
        "prepare_actor_geometric_occlusion_projection_context_export_inputs",
        prepare,
    )

    result = target.build_anchor_export_inputs(
        keyframe=keyframe,
        reader=object(),
    )

    assert result == ("prepared",)
    assert tuple(name for name, _ in calls) == ("build", "prepare")
    assert calls[0][1]["actors"] is actors
    assert calls[0][1]["cameras"] is cameras
    assert calls[1][1]["result"] is enhanced


def test_export_reuses_reader_per_clip_and_calls_v02_writer(monkeypatch, tmp_path):
    keyframes = [
        {"anchor_id": "a", "clip_id": "clip_1", "anchor_ns": 1},
        {"anchor_id": "b", "clip_id": "clip_1", "anchor_ns": 2},
        {"anchor_id": "c", "clip_id": "clip_2", "anchor_ns": 3},
    ]
    created = []
    prepared = []
    written = {}
    monkeypatch.setattr(target, "read_keyframes", lambda path: keyframes)

    def reader(path):
        value = SimpleNamespace(path=path)
        created.append(value)
        return value

    def build(*, keyframe, reader):
        value = (keyframe["anchor_id"], str(reader.path))
        prepared.append(value)
        return (value,)

    def write(**kwargs):
        written.update(kwargs)
        return {
            "row_count": 3,
            "evidence_status_counts": {},
            "rows_by_actor_class": {},
            "missing_surface_projection_context_count": 0,
            "missing_surface_projection_context_status_counts": {},
            "output_sha256": "test_digest",
        }

    monkeypatch.setattr(target, "DrivingClipReader", reader)
    monkeypatch.setattr(target, "build_anchor_export_inputs", build)
    monkeypatch.setattr(
        target,
        "write_actor_geometric_occlusion_projection_context_evidence",
        write,
    )

    target.export(
        keyframe_path=tmp_path / "keyframes.jsonl",
        output_path=tmp_path / "output.jsonl",
        summary_path=tmp_path / "summary.json",
    )

    assert len(created) == 2
    assert [item[0] for item in prepared] == ["a", "b", "c"]
    assert tuple(written["inputs"]) == tuple(prepared)


def test_v02_default_outputs_do_not_replace_v01_paths():
    assert target.OUTPUT_PATH.name.endswith("_v02.jsonl")
    assert target.SUMMARY_PATH.name.endswith("_v02.summary.json")
    assert "_v01" not in target.OUTPUT_PATH.name


def test_export_returns_writer_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(
        target,
        "read_keyframes",
        lambda path: [{"anchor_id": "a", "clip_id": "clip", "anchor_ns": 1}],
    )
    monkeypatch.setattr(target, "DrivingClipReader", lambda path: object())
    monkeypatch.setattr(
        target,
        "build_anchor_export_inputs",
        lambda **kwargs: (),
    )
    expected = {
        "row_count": 0,
        "evidence_status_counts": {},
        "rows_by_actor_class": {},
        "missing_surface_projection_context_count": 0,
        "missing_surface_projection_context_status_counts": {},
            "output_sha256": "test_digest",
    }
    monkeypatch.setattr(
        target,
        "write_actor_geometric_occlusion_projection_context_evidence",
        lambda **kwargs: expected,
    )

    result = target.export(
        keyframe_path=tmp_path / "keyframes.jsonl",
        output_path=tmp_path / "output.jsonl",
        summary_path=tmp_path / "summary.json",
    )

    assert result is expected
