#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_rebot_b601_follower import BiRebotB601Follower, BiRebotB601FollowerConfig
from lerobot.robots.rebot_b601_follower import (
    RebotB601Follower,
    RebotB601FollowerConfig,
    RebotB601FollowerRobotConfig,
)
from lerobot.robots.rebot_b601_follower.target_rate_limiter import JointTargetRateLimiter

_MODULE = "lerobot.robots.rebot_b601_follower.rebot_b601_follower"


def _make_motor_mock(position_rad: float = 0.0) -> MagicMock:
    motor = MagicMock(name="MotorMock")
    state = MagicMock()
    state.pos = position_rad
    motor.get_state.return_value = state
    return motor


def _make_bus_mock() -> MagicMock:
    bus = MagicMock(name="MotorBridgeControllerMock")
    # add_damiao_motor returns a fresh motor mock; position encodes the call order.
    bus._motor_count = 0

    def _add_motor(_send_id, _recv_id, _model):
        bus._motor_count += 1
        return _make_motor_mock(position_rad=math.radians(bus._motor_count))

    bus.add_damiao_motor.side_effect = _add_motor
    return bus


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_dm_serial.return_value = bus_mock
        cfg = RebotB601FollowerRobotConfig(port="/dev/null")
        robot = RebotB601Follower(cfg)
        robot.connect(calibrate=False)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_features_match_joints():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        robot = RebotB601Follower(RebotB601FollowerRobotConfig(port="/dev/null"))
    expected = {f"{m}.pos" for m in robot.motor_names}
    assert set(robot.action_features) == expected
    assert set(robot.observation_features) == expected
    assert "gripper.pos" in expected


def test_connect_disconnect(follower):
    assert follower.is_connected
    follower.disconnect()
    assert not follower.is_connected


def test_get_observation_converts_to_degrees(follower):
    obs = follower.get_observation()
    assert set(obs) == {f"{m}.pos" for m in follower.motor_names}
    # The bus mock seeds each motor's position with its 1-indexed creation order (radians).
    for idx, motor in enumerate(follower.motor_names, 1):
        assert obs[f"{motor}.pos"] == pytest.approx(math.degrees(math.radians(idx)))


def test_send_action_clips_to_joint_limits(follower):
    # shoulder_pan limit is (-150, 150); request beyond the upper bound.
    returned = follower.send_action({"shoulder_pan.pos": 999.0})
    assert returned["shoulder_pan.pos"] == 150.0
    # Default control_mode is "mit", so arm joints are driven via send_mit.
    follower.motors["shoulder_pan"].send_mit.assert_called_once()


def test_send_action_routes_gripper_to_force_pos(follower):
    follower.send_action({"gripper.pos": -10.0})
    follower.motors["gripper"].send_force_pos.assert_called_once()
    follower.motors["gripper"].send_pos_vel.assert_not_called()


def test_gripper_mit_mode_routes_to_send_mit():
    bus_mock = _make_bus_mock()
    with (
        patch(f"{_MODULE}.require_package", lambda *a, **kw: None),
        patch(f"{_MODULE}.MotorBridgeController") as controller_cls,
        patch(f"{_MODULE}.MotorBridgeMode", MagicMock()),
    ):
        controller_cls.from_dm_serial.return_value = bus_mock
        cfg = RebotB601FollowerRobotConfig(port="/dev/null", gripper_control_mode="mit")
        robot = RebotB601Follower(cfg)
        robot.connect(calibrate=False)
        robot.send_action({"gripper.pos": -10.0})
        robot.motors["gripper"].send_mit.assert_called_once()
        robot.motors["gripper"].send_force_pos.assert_not_called()


def test_bimanual_prefixes_features():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        cfg = BiRebotB601FollowerConfig(
            left_arm_config=RebotB601FollowerConfig(port="/dev/null0"),
            right_arm_config=RebotB601FollowerConfig(port="/dev/null1"),
        )
        robot = BiRebotB601Follower(cfg)
    assert any(k.startswith("left_") for k in robot.action_features)
    assert any(k.startswith("right_") for k in robot.action_features)
    assert "left_gripper.pos" in robot.action_features
    assert "right_gripper.pos" in robot.action_features


def test_bimanual_forwards_independent_target_rate_settings():
    with patch(f"{_MODULE}.require_package", lambda *a, **kw: None):
        cfg = BiRebotB601FollowerConfig(
            left_arm_config=RebotB601FollowerConfig(
                port="/dev/null0",
                max_relative_target=5.0,
                max_target_velocity_deg_s=30.0,
                max_target_step_deg=1.0,
            ),
            right_arm_config=RebotB601FollowerConfig(
                port="/dev/null1",
                max_relative_target=5.0,
                max_target_velocity_deg_s=60.0,
                max_target_step_deg=2.0,
            ),
        )
        robot = BiRebotB601Follower(cfg)
    assert robot.left_arm._target_limiter.velocity_deg_s == 30
    assert robot.right_arm._target_limiter.velocity_deg_s == 60
    assert robot.left_arm._target_limiter.step_deg == 1
    assert robot.right_arm._target_limiter.step_deg == 2


def test_driver_advances_reference_without_changing_units_or_gains(follower):
    follower._target_limiter = JointTargetRateLimiter(30, 1)
    follower.config.max_relative_target = 5.0
    with patch(
        "lerobot.robots.rebot_b601_follower.rebot_b601_follower.time.monotonic",
        side_effect=[0, 1 / 30, 2 / 30],
    ):
        sent = [follower.send_action({"shoulder_pan.pos": 100.0}) for _ in range(3)]
    assert [s["shoulder_pan.pos"] for s in sent] == pytest.approx([1, 2, 3])
    follower.motors["shoulder_pan"].send_mit.assert_called_with(math.radians(3), 0.0, 45.0, 12.0, 0.0)


def test_driver_missing_feedback_and_failed_send_cannot_resume_silently(follower):
    follower._target_limiter = JointTargetRateLimiter(30, 1)
    follower.config.max_relative_target = 5.0
    motor = follower.motors["shoulder_pan"]
    state = motor.get_state.return_value
    motor.get_state.return_value = None
    with pytest.raises(ValueError, match="Missing"):
        follower.send_action({"shoulder_pan.pos": 100.0})
    motor.send_mit.assert_not_called()
    motor.get_state.return_value = state
    motor.send_mit.side_effect = OSError("send failed")
    with pytest.raises(OSError):
        follower.send_action({"shoulder_pan.pos": 100.0})
    with pytest.raises(RuntimeError, match="reconnect"):
        follower.send_action({"shoulder_pan.pos": 100.0})


def test_rate_only_driver_can_advance_beyond_old_error_and_software_angle_bounds(follower):
    follower._target_limiter = JointTargetRateLimiter(60, 2)
    follower.config.max_relative_target = None
    follower.config.joint_limits = {}
    with patch(f"{_MODULE}.time.monotonic", side_effect=[i / 30 for i in range(101)]):
        sent = [follower.send_action({"shoulder_pan.pos": 1000.0})["shoulder_pan.pos"] for _ in range(101)]
    assert sent[0] == pytest.approx(1)
    assert sent[-1] == pytest.approx(201)
    assert max(b - a for a, b in zip(sent, sent[1:], strict=False)) <= 2 + 1e-9


def test_rate_limited_driver_rejects_nonfinite_action_before_soft_clipping(follower):
    follower._target_limiter = JointTargetRateLimiter(60, 2)
    with pytest.raises(ValueError, match="Nonfinite"):
        follower.send_action({"shoulder_pan.pos": math.nan})
    follower.motors["shoulder_pan"].send_mit.assert_not_called()
