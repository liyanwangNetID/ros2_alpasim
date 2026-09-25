from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from project_paths import ALPASIM_DATA_ROOT

@dataclass(frozen=True)
class Step9Config:
    data_root: Path = ALPASIM_DATA_ROOT
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:14b"
    expected_family: str = "qwen3"
    expected_parameter_size: str = "14.8B"
    expected_quantization: str = "Q4_K_M"
    num_ctx: int = 4096
    num_predict: int = 384
    temperature: float = 0.0
    seed: int = 9
    keep_alive: str = "30m"
    max_attempts: int = 3
    timeout_seconds: float = 180.0

    @classmethod
    def from_environment(cls, data_root: Path | str | None = None) -> "Step9Config":
        return cls(
            data_root=Path(data_root or os.environ.get("ALPASIM_DATA_ROOT", ALPASIM_DATA_ROOT)).expanduser().resolve(),
            base_url=os.environ.get("STEP9_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
            model=os.environ.get("STEP9_OLLAMA_MODEL", "qwen3:14b"),
        )

    @property
    def input_path(self) -> Path:
        return self.data_root / "annotations" / "v0.1-draft" / "structured_causality.jsonl"

    @property
    def input_contract_path(self) -> Path:
        return self.data_root / "manifests" / "structured_causality_contract_v0.1.json"

    @property
    def output_path(self) -> Path:
        return self.data_root / "annotations" / "v0.1-draft" / "reasoning.jsonl"

    @property
    def schema_path(self) -> Path:
        return self.data_root / "schemas" / "reasoning_schema_v0.1-draft.json"

    @property
    def summary_path(self) -> Path:
        return self.data_root / "reports" / "step9_reasoning_summary_v01.json"

    @property
    def contract_path(self) -> Path:
        return self.data_root / "manifests" / "reasoning_contract_v0.1.json"

    @property
    def work_root(self) -> Path:
        return self.data_root / "reports" / "step9_local_llm"

    @property
    def checkpoint_path(self) -> Path:
        return self.work_root / "accepted.jsonl"

    @property
    def rejection_path(self) -> Path:
        return self.work_root / "rejected.jsonl"

    @property
    def run_manifest_path(self) -> Path:
        return self.work_root / "run_manifest.json"
