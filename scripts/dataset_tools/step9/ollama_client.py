from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Mapping

from .config import Step9Config
from .contract import response_schema
from .prompt import SYSTEM_PROMPT, user_prompt


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, config: Step9Config):
        self.config = config

    def _post(self, endpoint: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self.config.base_url + endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                value = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc
        if not isinstance(value, dict):
            raise OllamaError("Ollama response must be an object")
        return value

    def model_metadata(self) -> dict[str, Any]:
        return self._post("/api/show", {"model": self.config.model})

    def validate_model(self) -> dict[str, Any]:
        metadata = self.model_metadata()
        details = metadata.get("details") or {}
        thinking = metadata.get("thinking") or {}
        expected = {
            "family": self.config.expected_family,
            "parameter_size": self.config.expected_parameter_size,
            "quantization_level": self.config.expected_quantization,
        }
        mismatches = {
            key: (expected[key], details.get(key))
            for key in expected
            if details.get(key) != expected[key]
        }
        if mismatches:
            raise OllamaError(f"unexpected model metadata: {mismatches}")
        if False not in thinking.get("values", []):
            raise OllamaError("model does not support think=false")
        return metadata

    def generate(
        self,
        package: Mapping[str, Any],
        *,
        attempt: int = 1,
        validation_feedback: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(package)},
        ]
        if validation_feedback:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous answer failed validation: "
                        + validation_feedback
                        + ". Regenerate from the original evidence package. "
                        "Do not mention a longitudinal action unless a supports relation "
                        "explicitly targets longitudinal_decision. Return only the JSON object."
                    ),
                }
            )
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": response_schema(),
            "options": {
                "temperature": self.config.temperature,
                "seed": self.config.seed + max(0, attempt - 1),
                "num_ctx": self.config.num_ctx,
                "num_predict": self.config.num_predict,
            },
            "keep_alive": self.config.keep_alive,
        }
        envelope = self._post("/api/chat", payload)
        if not envelope.get("done"):
            raise OllamaError("Ollama response is incomplete")
        content = (envelope.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise OllamaError("Ollama response has no message.content")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"message.content is not JSON: {exc}") from exc
        if not isinstance(result, dict):
            raise OllamaError("message.content JSON must be an object")
        return result, envelope
