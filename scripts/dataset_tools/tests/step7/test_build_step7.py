from __future__ import annotations

from types import SimpleNamespace

import pytest

from step7 import build_step7 as target


def test_stage_order_matches_step_dependencies():
    assert tuple(stage.name for stage in target.STAGES) == (
        "projection_evidence",
        "occlusion",
        "observability_policy",
        "history",
        "road_context",
        "actor_roles",
        "scene_features",
        "scene_facts",
    )


def test_select_stages_supports_inclusive_range_and_skip():
    selected = target.select_stages(
        from_stage="history",
        to_stage="scene_features",
        skipped=("road_context",),
    )
    assert tuple(stage.name for stage in selected) == (
        "history",
        "actor_roles",
        "scene_features",
    )


def test_select_stages_rejects_reverse_range():
    with pytest.raises(ValueError, match="from-stage"):
        target.select_stages(
            from_stage="scene_facts",
            to_stage="history",
            skipped=(),
        )


def test_dry_run_does_not_create_subprocess(monkeypatch, capsys):
    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called")

    monkeypatch.setattr(target.subprocess, "run", fail_run)
    elapsed = target.run_stage(
        target.STAGES[0],
        position=1,
        total=1,
        dry_run=True,
    )
    assert elapsed == 0.0
    assert "dry-run command" in capsys.readouterr().out


def test_stage_failure_is_propagated(monkeypatch):
    monkeypatch.setattr(
        target.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=7),
    )
    with pytest.raises(RuntimeError, match="exit code 7"):
        target.run_stage(
            target.STAGES[0],
            position=1,
            total=1,
            dry_run=False,
        )


def test_main_dry_run_selected_tail(capsys):
    assert target.main(
        (
            "--from-stage",
            "scene_features",
            "--dry-run",
        )
    ) == 0
    output = capsys.readouterr().out
    assert "scene_features" in output
    assert "scene_facts" in output
    assert "projection_evidence" not in output
    assert "PASS: unified Step 7 build completed." in output
