from __future__ import annotations

import pytest

from step8.contract import JoinedAnchor, Step8ContractError, join_rows


def row(anchor_id: str, clip_id: str = "clip", anchor_ns: int = 10):
    return {"anchor_id": anchor_id, "clip_id": clip_id, "anchor_ns": anchor_ns}


def test_join_allows_extra_candidate_meta_actions():
    keyframes = {"a": row("a"), "b": row("b", anchor_ns=20)}
    navigation = {"a": row("a"), "b": row("b", anchor_ns=20)}
    scenes = {"a": row("a"), "b": row("b", anchor_ns=20)}
    meta = {
        "a": row("a"),
        "b": row("b", anchor_ns=20),
        "candidate_only": row("candidate_only", anchor_ns=30),
    }
    joined = join_rows(["a", "b"], keyframes, navigation, scenes, meta)
    assert [item.anchor_id for item in joined] == ["a", "b"]
    assert all(isinstance(item, JoinedAnchor) for item in joined)


def test_join_rejects_missing_meta_action():
    rows = {"a": row("a")}
    with pytest.raises(Step8ContractError, match="do not cover"):
        join_rows(["a"], rows, rows, rows, {})


def test_join_rejects_extra_navigation_anchor():
    rows = {"a": row("a")}
    navigation = {"a": row("a"), "x": row("x")}
    with pytest.raises(Step8ContractError, match="coverage mismatch"):
        join_rows(["a"], rows, navigation, rows, rows)


def test_join_rejects_identity_mismatch():
    rows = {"a": row("a")}
    navigation = {"a": row("a", clip_id="other")}
    with pytest.raises(Step8ContractError, match="identity mismatch"):
        join_rows(["a"], rows, navigation, rows, rows)


def test_join_preserves_keyframe_order():
    keyframes = {"b": row("b", anchor_ns=20), "a": row("a")}
    navigation = dict(keyframes)
    scenes = dict(keyframes)
    meta = dict(keyframes)
    joined = join_rows(["b", "a"], keyframes, navigation, scenes, meta)
    assert [item.anchor_id for item in joined] == ["b", "a"]
