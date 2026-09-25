import json

from step9.config import Step9Config
from step9.ollama_client import OllamaClient


def package():
    return {
        "record_id": "a",
        "decisions": {},
        "quality_status": "usable",
        "quality_reasons": [],
        "evidence": [],
    }


def test_retry_adds_feedback_and_changes_seed(monkeypatch):
    client = OllamaClient(Step9Config())
    captured = {}

    def fake_post(endpoint, payload):
        captured.update(payload)
        return {
            "done": True,
            "message": {
                "content": json.dumps(
                    {
                        "reasoning_summary": "Limited evidence.",
                        "used_evidence_keys": [],
                        "limitations": [],
                    }
                )
            },
        }

    monkeypatch.setattr(client, "_post", fake_post)
    client.generate(
        package(),
        attempt=2,
        validation_feedback="unsupported longitudinal claim",
    )
    assert captured["options"]["seed"] == 10
    assert len(captured["messages"]) == 3
    assert "failed validation" in captured["messages"][2]["content"]
