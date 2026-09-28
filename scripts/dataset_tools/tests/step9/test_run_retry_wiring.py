from pathlib import Path


def test_production_run_passes_retry_state_to_client():
    source = (Path(__file__).parents[2] / "step9" / "run.py").read_text(encoding="utf-8")
    compact = "".join(source.split())
    expected = "client.generate(package,attempt=attempt,validation_feedback=last_error)"
    assert expected in compact
