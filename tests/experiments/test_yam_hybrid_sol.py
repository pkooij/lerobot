"""The hybrid pilot uses bounded Cartesian IK and calibrated grippers."""

from dataclasses import asdict

import draccus
import pytest

from examples.yam_experiments.hybrid_sol import checked_pose, hybrid_config, ik_check, planner_config
from lerobot.rollout.hybrid import HybridConfig, PlannerDecision


def test_pilot_contract_covers_both_arms_and_disables_raw_joint_changes():
    config = hybrid_config()
    assert len(config.limits) == 14
    decoded = draccus.decode(HybridConfig, asdict(config))
    for key, limit in decoded.limits.items():
        assert limit.max_delta == (1 if "gripper" in key else 0.25)
    pose = dict.fromkeys(config.limits, 0.0)
    correction = PlannerDecision(
        "intervention", "Visible object", "Adjust", "", {"left_joint_0.pos": 0.01}, 1
    )
    with pytest.raises(ValueError, match="Use end_effector"):
        correction.validate_motion(config, pose)
    correction = PlannerDecision("intervention", "Over bin", "Release", "", {"left_gripper.pos": 1}, 1)
    assert correction.validate_motion(config, pose)["left_gripper.pos"] == 1


def test_sol_client_requires_environment_credential(tmp_path, monkeypatch):
    monkeypatch.delenv("SOL_SERVICE_TIER", raising=False)
    config = planner_config(tmp_path / "planner.jsonl")
    assert config.model_id == "gpt-6.1-sol"
    assert config.api_key_env == "OPENAI_API_KEY"
    assert config.api_key == "EMPTY"
    assert config.api_mode == "responses"
    assert config.request_max_retries == 0
    assert config.service_tier == "fast"
    monkeypatch.setenv("SOL_SERVICE_TIER", "default")
    assert planner_config(tmp_path / "planner.jsonl").service_tier == "default"


def test_yam_tool_transform_and_ik_round_trips():
    pytest.importorskip("mujoco")
    from lerobot.rollout.end_effector import EndEffectorKinematics, parse_pose

    config = hybrid_config()
    pose = dict.fromkeys(config.limits, 0.0)
    report = ik_check(config, pose)
    assert not report["executed"]
    for name, ee in config.end_effectors.items():
        solver = EndEffectorKinematics(ee)
        position, rotation = parse_pose(solver.forward(pose))
        # Independently composed I2RT yam/v1 + linear_4310, pinned 120c3c814.
        # Catches omission of the gripper attachment rotation (a 29 cm TCP error).
        assert position == pytest.approx([0.2552479, -0.000038398, 0.17340419], abs=1e-9)
        assert rotation.apply([0, 0, 1]) == pytest.approx([1, 0, 0], abs=1e-9)
        assert rotation.apply([1, 0, 0]) == pytest.approx([0, 0, -1], abs=1e-9)
        assert len(report["end_effectors"][name]["round_trips"]) == 6
        assert (
            max(check["position_error_m"] for check in report["end_effectors"][name]["round_trips"]) < 0.001
        )
        assert solver.forward(pose | {f"{name}_gripper.pos": 1}) == solver.forward(pose)


def test_preflight_pose_uses_runtime_feedback_tolerance():
    config = hybrid_config()
    pose = dict.fromkeys(config.limits, 0.0)
    pose["left_joint_1.pos"] = -0.00019073777370870462
    assert checked_pose(config, pose)["left_joint_1.pos"] == 0
    pose["left_joint_1.pos"] = -0.1
    with pytest.raises(ValueError, match="feedback tolerance"):
        checked_pose(config, pose)


def test_estimated_wrist_mount_has_correct_zero_pose_axes():
    import numpy as np

    pytest.importorskip("mujoco")
    from lerobot.rollout.end_effector import EndEffectorKinematics, parse_pose

    config = hybrid_config()
    pose = dict.fromkeys(config.limits, 0.0)
    for side, ee in config.end_effectors.items():
        solver = EndEffectorKinematics(ee)
        tip, _ = parse_pose(solver.forward(pose))
        camera = solver.camera_poses(pose)[side]
        transform = np.asarray(camera["T_base_from_camera"])
        # At zero joints: camera behind (-base X), above (+base Z), looking forward/down.
        assert transform[:3, 3] - tip == pytest.approx([-0.075, 0, 0.070], abs=1e-10)
        assert transform[:3, 2] == pytest.approx([np.cos(np.deg2rad(25)), 0, -np.sin(np.deg2rad(25))])
        assert transform[:3, 0] == pytest.approx([0, -1, 0], abs=1e-10)
        assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)
        assert camera["estimated"]
