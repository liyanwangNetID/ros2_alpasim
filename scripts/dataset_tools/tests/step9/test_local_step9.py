import json
from pathlib import Path
import pytest
from step9.config import Step9Config
from step9.contract import source_record_sha256
from step9.prompt import build_evidence_package, evidence_key
from step9.validator import validate_response
from step9.checkpoint import append_jsonl, load_checkpoint
from step9.ollama_client import OllamaClient, OllamaError

def source(quality="usable"):
    return {"anchor_id":"test_clip_001_1","clip_id":"test_clip_001","anchor_ns":1,"quality":{"status":quality,"reasons":[] if quality=="usable" else ["navigation_quality_unknown"],"source_quality":{"navigation":"usable","scene_facts":"usable","meta_action":"usable"}},"structured_coc":{"nodes":[{"node_id":"decision_lateral","node_type":"lateral_decision","value":{"action":"keep_direction","quality_status":"usable"},"source_ref":"x"},{"node_id":"decision_longitudinal","node_type":"longitudinal_decision","value":{"action":"maintain_speed","quality_status":"usable"},"source_ref":"x"},{"node_id":"navigation_intent","node_type":"navigation_intent","value":{"action":"straight","quality_status":"usable"},"source_ref":"x"}],"links":[{"link_id":"link_001","source_node_id":"navigation_intent","target_node_id":"decision_lateral","relation":"aligns_with","confidence":"supported","rule_id":"navigation_lateral_alignment_v01","reasons":["compatible"]}]}}

def test_package_removes_provenance_and_gives_evidence_key():
    package=build_evidence_package(source())
    assert package["record_id"]=="test_clip_001_1"
    assert package["evidence"][0]["evidence_key"]=="navigation_alignment_01"
    assert "link_id" not in json.dumps(package)

def test_validator_usable():
    package=build_evidence_package(source())
    value={"reasoning_summary":"The straight navigation action is compatible with keeping direction.","used_evidence_keys":["navigation_alignment_01"],"limitations":[]}
    assert validate_response(value,package)==value

def test_validator_partial_requires_explicit_uncertainty():
    package=build_evidence_package(source("partial"))
    with pytest.raises(ValueError): validate_response({"reasoning_summary":"Keep direction.","used_evidence_keys":[],"limitations":["Navigation quality is unknown."]},package)
    value={"reasoning_summary":"Navigation evidence is insufficient to explain the lateral decision.","used_evidence_keys":[],"limitations":["Navigation quality is unknown."]}
    assert validate_response(value,package)==value

def test_validator_rejects_unknown_key():
    with pytest.raises(ValueError): validate_response({"reasoning_summary":"Compatible.","used_evidence_keys":["made_up_01"],"limitations":[]},build_evidence_package(source()))

def test_checkpoint_round_trip(tmp_path):
    path=tmp_path/"accepted.jsonl"; row={"anchor_id":"a","value":1}; append_jsonl(path,row); assert load_checkpoint(path)=={"a":row}

def test_source_digest_is_stable(): assert source_record_sha256(source())==source_record_sha256(dict(reversed(list(source().items()))))

def test_config_paths(tmp_path):
    config=Step9Config(data_root=tmp_path); assert config.input_path==tmp_path/"annotations"/"v0.1-draft"/"structured_causality.jsonl"

def test_model_validation(monkeypatch):
    client=OllamaClient(Step9Config())
    monkeypatch.setattr(client,"model_metadata",lambda:{"details":{"family":"qwen3","parameter_size":"14.8B","quantization_level":"Q4_K_M"},"thinking":{"values":[False,True]}})
    assert client.validate_model()["details"]["family"]=="qwen3"

def test_model_validation_rejects_quantization(monkeypatch):
    client=OllamaClient(Step9Config())
    monkeypatch.setattr(client,"model_metadata",lambda:{"details":{"family":"qwen3","parameter_size":"14.8B","quantization_level":"Q8"},"thinking":{"values":[False,True]}})
    with pytest.raises(OllamaError): client.validate_model()
