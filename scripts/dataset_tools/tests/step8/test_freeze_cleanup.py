from pathlib import Path

from step8 import build_step8
from step8.semantic_profile import append_unique_example


def test_semantic_profile_examples_are_unique_and_limited():
    values = []
    for anchor in ["a", "a", "b", "c", "d"]:
        append_unique_example(values, anchor, 3)
    assert values == ["a", "b", "c"]


def test_build_main_reports_existing_outputs_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        build_step8.Step8Paths,
        "from_data_root",
        classmethod(lambda cls, value=None: object()),
    )
    monkeypatch.setattr(
        build_step8,
        "build",
        lambda paths, force: (_ for _ in ()).throw(
            FileExistsError("Step 8 outputs already exist; use --force")
        ),
    )
    monkeypatch.setattr("sys.argv", ["build_step8"])
    assert build_step8.main() == 2
    captured = capsys.readouterr()
    assert "ERROR: Step 8 outputs already exist; use --force" in captured.err
    assert "Traceback" not in captured.err


def test_obsolete_inventory_module_is_removed():
    root = Path(__file__).resolve().parents[2]
    assert not (root / "step8" / "inventory.py").exists()
