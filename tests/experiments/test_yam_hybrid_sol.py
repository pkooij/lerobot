"""The first hybrid pilot permits calibrated grippers, not uncommissioned arm corrections."""

from dataclasses import asdict

import draccus
import pytest

from examples.yam_experiments.hybrid_sol import hybrid_config, planner_config
from lerobot.rollout.hybrid import HybridConfig, PlannerDecision


def test_pilot_contract_covers_both_arms_and_disables_joint_changes():
    config = hybrid_config()
    assert len(config.limits) == 14
    decoded = draccus.decode(HybridConfig, asdict(config))
    for key, limit in decoded.limits.items():
        assert limit.max_delta == (1 if "gripper" in key else 0)
    pose = dict.fromkeys(config.limits, 0.0)
    correction = PlannerDecision(
        "intervention", "Visible object", "Adjust", "", {"left_joint_0.pos": 0.01}, 1
    )
    with pytest.raises(ValueError, match="delta/speed"):
        correction.validate_motion(config, pose)
    correction = PlannerDecision("intervention", "Over bin", "Release", "", {"left_gripper.pos": 1}, 1)
    assert correction.validate_motion(config, pose)["left_gripper.pos"] == 1


def test_sol_client_requires_environment_credential(tmp_path):
    config = planner_config(tmp_path / "planner.jsonl")
    assert config.model_id == "gpt-6.1-sol"
    assert config.api_key_env == "OPENAI_API_KEY"
    assert config.api_key == "EMPTY"
    assert config.api_mode == "responses"
    assert config.request_max_retries == 0
