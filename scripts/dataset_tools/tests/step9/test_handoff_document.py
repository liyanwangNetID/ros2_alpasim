from pathlib import Path


def test_step9_handoff_is_canonical_and_complete():
    workspace = Path(__file__).resolve().parents[4]
    handoff = workspace / "docs" / "AI_DATASET_DEVELOPMENT_HANDOFF.md"
    step9_readme = workspace / "scripts" / "dataset_tools" / "step9" / "README.md"
    assert handoff.is_file()
    assert not step9_readme.exists()
    text = handoff.read_text(encoding="utf-8")
    required = (
        "Step 9: Human-readable reasoning",
        "qwen3:14b",
        "14.8B",
        "Q4_K_M",
        "step9_local_reasoning_prompt_v0.2",
        "b65a2d068b46c97bf1a0721b5a1078cac6f18d24e9fc4c8b89256a3b4de9a8af",
        "cd9b8441dd6dc9504a6f0c4b8a3406c2de4400f5c2c2533c36e24384a26a4b26",
        "default_training_usage: excluded",
        "optional_experimental_usage: auxiliary_only",
        "continue directly with Step 10",
    )
    for value in required:
        assert value in text
