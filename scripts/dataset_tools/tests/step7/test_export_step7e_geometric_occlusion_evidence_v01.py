import json
from types import SimpleNamespace

import pytest

from step7 import export_step7e_geometric_occlusion_evidence_v01 as target


def test_read_keyframes_validates_and_sorts(tmp_path):
    path = tmp_path / "keyframes.jsonl"
    path.write_text(
        json.dumps({"anchor_id": "b", "clip_id": "clip_b", "anchor_ns": 2})
        + "\n"
        + json.dumps({"anchor_id": "a", "clip_id": "clip_a", "anchor_ns": 1})
        + "\n"
    )

    result = target.read_keyframes(path)

    assert [item["anchor_id"] for item in result] == ["a", "b"]


def test_duplicate_keyframe_anchor_is_rejected(tmp_path):
    path = tmp_path / "keyframes.jsonl"
    row = {"anchor_id": "a", "clip_id": "clip", "anchor_ns": 1}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with pytest.raises(ValueError, match="must be unique"):
        target.read_keyframes(path)


def test_export_reuses_reader_per_clip_and_calls_writer(monkeypatch, tmp_path):
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
        return SimpleNamespace(
            row_count=3,
            evidence_status_counts=(),
            rows_by_actor_class=(),
        )

    monkeypatch.setattr(target, "DrivingClipReader", reader)
    monkeypatch.setattr(target, "build_anchor_export_inputs", build)
    monkeypatch.setattr(
        target,
        "write_actor_geometric_occlusion_evidence",
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


def test_exact_inputs_requires_exact_ego():
    reader = SimpleNamespace(
        get_recorded_ego_state_at_or_before=lambda value: SimpleNamespace(
            stamp_ns=value - 1
        ),
        get_actor_snapshots=lambda *args, **kwargs: (),
    )

    with pytest.raises(RuntimeError, match="Exact recorded Ego"):
        target.exact_inputs(reader, 100)
