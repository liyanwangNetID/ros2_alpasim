from step3.candidate_anchors import (
    ANCHOR_FORMAT_VERSION,
    CONTRACT_VERSION,
    SCRIPT_VERSION,
    candidate_anchor_contract,
)


def test_candidate_anchor_contract_uses_generated_values():
    anchor_text = (
        '{"anchor_id":"a"}\n'
        '{"anchor_id":"b"}\n'
    )
    summary = {
        "clips": {
            "processed": 4,
            "with_selected_anchors": 3,
            "without_selected_anchors": 1,
        },
        "anchors": {
            "selected_total": 2,
        },
    }

    contract = candidate_anchor_contract(
        anchor_text=anchor_text,
        summary=summary,
    )

    assert contract["contract_version"] == CONTRACT_VERSION
    assert contract["producer_step"] == 3
    assert (
        contract["anchor_format_version"]
        == ANCHOR_FORMAT_VERSION
    )
    assert (
        contract["anchor_selector_version"]
        == SCRIPT_VERSION
    )
    assert contract["candidate_anchor_count"] == 2
    assert contract["processed_clip_count"] == 4
    assert contract["clips_with_selected_anchors"] == 3
    assert contract["clips_without_selected_anchors"] == 1
    assert len(contract["candidate_anchor_sha256"]) == 64


def test_candidate_anchor_contract_is_deterministic():
    summary = {
        "clips": {
            "processed": 1,
            "with_selected_anchors": 1,
            "without_selected_anchors": 0,
        },
        "anchors": {
            "selected_total": 1,
        },
    }

    first = candidate_anchor_contract(
        anchor_text='{"anchor_id":"a"}\n',
        summary=summary,
    )
    second = candidate_anchor_contract(
        anchor_text='{"anchor_id":"a"}\n',
        summary=summary,
    )

    assert first == second
