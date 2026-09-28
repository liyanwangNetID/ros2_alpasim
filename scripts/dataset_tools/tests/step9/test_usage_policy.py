from step9.contract import usage_policy


def test_step9_usage_policy_is_non_authoritative_and_excluded_by_default():
    policy = usage_policy()
    assert policy["authoritative"] is False
    assert policy["default_training_usage"] == "excluded"
    assert policy["human_review"] is True
    assert policy["sample_acceptance_dependency"] is False
    assert policy["action_target_dependency"] is False
    assert policy["dataset_split_dependency"] is False
    assert policy["optional_experimental_usage"] == "auxiliary_only"
    assert policy["source_of_truth_step"] == 8
